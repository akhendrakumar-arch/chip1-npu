/*
 * tt_um_akhendrakumar_npu
 * Reconfigurable INT8 MLP inference accelerator (weight-streaming),
 * scaled to a fully runtime-configurable shape.
 *
 * Weights live off-chip in the RP2040 (spi-ram-emu). This chip is the
 * SPI master: it issues a read command + 24-bit address and shifts in
 * INT8 weight bytes. Activations are streamed in over ui_in. The chip
 * does INT8 x INT8 -> INT32 MACs, ReLU, and returns an argmax class.
 *
 * The datapath is time-multiplexed: one MAC unit is folded over every
 * neuron and every layer, so area stays roughly constant regardless of
 * shape. "Reconfigurable" = same silicon, different weights + config:
 * the number of weight-matrix layers (MAX_LAYERS ceiling) and the
 * width of every layer, including the input (MAX_LAYER_N ceiling),
 * are set entirely at runtime via config bytes -- nothing about the
 * network shape is compiled in beyond the two ceilings below.
 */

`default_nettype none
/* verilator lint_off WIDTH */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off DECLFILENAME */

module tt_um_akhendrakumar_npu (
    input  wire [7:0] ui_in,    // dedicated inputs  - activation/config bus
    output wire [7:0] uo_out,   // dedicated outputs - result
    input  wire [7:0] uio_in,   // bidir: input path
    output wire [7:0] uio_out,  // bidir: output path
    output wire [7:0] uio_oe,   // bidir: output enable (1 = drive)
    input  wire       ena,      // always 1 when powered
    input  wire       clk,      // system clock
    input  wire       rst_n     // active-low reset
);

    // ---- pin mapping (matches info.yaml) ----
    // uio[0] spi_sclk (out)  uio[1] spi_cs_n (out)  uio[2] spi_mosi (out)
    // uio[3] spi_miso (in)   uio[4] start (in)      uio[5] busy_done (out)
    // uio[6] mode_sel (in)   uio[7] strobe (in)
    wire        spi_miso  = uio_in[3];
    wire        start     = uio_in[4];
    wire        mode_sel  = uio_in[6];  // 0 = config load, 1 = data load
    wire        strobe    = uio_in[7];

    reg  spi_sclk, spi_cs_n, spi_mosi, busy_done;

    // Drive only the output-role bidir pins; per-bit per info.yaml:
    //   uio[0] sclk, uio[1] cs_n, uio[2] mosi, uio[5] busy_done.
    wire [7:0] uio_out_i;
    assign uio_out_i[0] = spi_sclk;
    assign uio_out_i[1] = spi_cs_n;
    assign uio_out_i[2] = spi_mosi;
    assign uio_out_i[3] = 1'b0;
    assign uio_out_i[4] = 1'b0;
    assign uio_out_i[5] = busy_done;
    assign uio_out_i[6] = 1'b0;
    assign uio_out_i[7] = 1'b0;

    assign uio_oe = 8'b0010_0111; // drive sclk,cs_n,mosi,busy_done -> bits 0,1,2,5

    // ---- hardware ceilings for the runtime-configurable shape ----
    localparam MAX_LAYERS  = 3;  // weight-matrix layers: in->L1->L2->out is 3
    localparam MAX_LAYER_N = 8;  // widest any single layer (incl. the input) may be

    // ---- config registers (loaded when mode_sel=0, on strobe) ----
    // cfg_idx cycles: 0=layer_count, 1..4=layer_size[0..3], 5=act_sel
    reg [1:0] layer_count;                // number of weight-matrix layers, 1..MAX_LAYERS
    reg [3:0] layer_size [0:MAX_LAYERS];  // [0]=input width, [1..MAX_LAYERS]=each layer's output width
    reg       act_sel;                    // 0 = ReLU on hidden layers, 1 = identity
    reg [2:0] cfg_idx;

    // ---- neuron value storage: ping-pong buffers so an arbitrary ----
    // ---- number of layers can be folded over one pair of arrays  ----
    reg signed [7:0] buf0 [0:MAX_LAYER_N-1];
    reg signed [7:0] buf1 [0:MAX_LAYER_N-1];
    reg        cur_sel;   // 0: current=buf0/next=buf1   1: current=buf1/next=buf0
    reg [3:0]  act_idx;

    // ---- SPI master: read one weight byte by address ----
    // Simplified spi-ram-emu style: assert cs, shift 8-bit read cmd +
    // 24-bit addr, then shift 8 data bits in on miso. One byte per call.
    reg  [2:0]  spi_state;
    reg  [5:0]  spi_bitcnt;
    reg  [9:0]  spi_addr;       // low 10 bits of the 24-bit address; bits above are always 0
    reg  signed [7:0] spi_data;
    reg         spi_go, spi_ready;
    localparam SPI_IDLE=0, SPI_CMD=1, SPI_ADDR=2, SPI_DATA=3, SPI_DONE=4;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spi_state<=SPI_IDLE; spi_sclk<=0; spi_cs_n<=1; spi_mosi<=0;
            spi_bitcnt<=0; spi_data<=0; spi_ready<=0;
        end else begin
            spi_ready<=0;
            case (spi_state)
                SPI_IDLE: begin
                    spi_cs_n<=1; spi_sclk<=0;
                    if (spi_go) begin
                        spi_cs_n<=0; spi_bitcnt<=8; spi_state<=SPI_CMD;
                    end
                end
                SPI_CMD: begin // shift out read opcode 0x03
                    spi_mosi <= 8'h03 >> (spi_bitcnt-1);
                    spi_sclk <= ~spi_sclk;
                    if (spi_sclk) begin
                        spi_bitcnt<=spi_bitcnt-1;
                        if (spi_bitcnt==1) begin spi_bitcnt<=24; spi_state<=SPI_ADDR; end
                    end
                end
                SPI_ADDR: begin // shift out 24-bit address (top 14 bits are always 0)
                    spi_mosi <= (spi_bitcnt > 10) ? 1'b0 : spi_addr[spi_bitcnt-1];
                    spi_sclk <= ~spi_sclk;
                    if (spi_sclk) begin
                        spi_bitcnt<=spi_bitcnt-1;
                        if (spi_bitcnt==1) begin spi_bitcnt<=8; spi_state<=SPI_DATA; end
                    end
                end
                SPI_DATA: begin // shift in 8 data bits on miso
                    spi_sclk <= ~spi_sclk;
                    if (spi_sclk) begin
                        spi_data <= {spi_data[6:0], spi_miso};
                        spi_bitcnt<=spi_bitcnt-1;
                        if (spi_bitcnt==1) spi_state<=SPI_DONE;
                    end
                end
                SPI_DONE: begin
                    spi_cs_n<=1; spi_sclk<=0; spi_ready<=1; spi_state<=SPI_IDLE;
                end
                default: spi_state<=SPI_IDLE;
            endcase
        end
    end

    // ---- MLP control FSM ----
    reg  [3:0]  st;
    reg  [1:0]  layer_idx;      // which weight-matrix layer we're computing, 0..MAX_LAYERS-1
    reg  [3:0]  n;              // current output neuron within this layer
    reg  [3:0]  i;              // current input index within this layer
    reg  signed [18:0] acc;     // max |sum of MAX_LAYER_N int8*int8 terms| = 8*16384 = 131072, fits in 19 bits
    reg  [3:0]  best_idx;
    reg  signed [18:0] best_val;
    reg  [9:0]  layer_base;     // running weight-table base address for the current layer
    localparam S_IDLE=0,S_FETCH=1,S_WAIT=2,S_MAC=3,S_ACT=4,S_NEXT=5,
               S_ARG=6,S_DONE=7;

    function signed [7:0] relu; input signed [18:0] v;
        relu = (v[18]) ? 8'sd0 :
               (v > 19'sd127) ? 8'sd127 : v[7:0];
    endfunction

    reg signed [7:0] cur_in;    // current operand from prev layer / input

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st<=S_IDLE; layer_idx<=0; n<=0; i<=0; acc<=0; spi_go<=0;
            busy_done<=0; best_idx<=0; best_val<=0; spi_addr<=0;
            layer_count<=1;
            layer_size[0]<=4'd8; layer_size[1]<=4'd8;
            layer_size[2]<=4'd8; layer_size[3]<=4'd8;
            act_sel<=0; cfg_idx<=0; act_idx<=0; cur_sel<=0; layer_base<=0;
        end else begin
            spi_go<=0;
            // ---- config / activation loading (only when idle) ----
            if (st==S_IDLE) begin
                busy_done<=0;
                if (strobe && !mode_sel) begin
                    case (cfg_idx)
                        0: layer_count   <= ui_in[1:0];
                        1: layer_size[0] <= ui_in[3:0];
                        2: layer_size[1] <= ui_in[3:0];
                        3: layer_size[2] <= ui_in[3:0];
                        4: layer_size[3] <= ui_in[3:0];
                        5: act_sel       <= ui_in[0];
                    endcase
                    cfg_idx <= (cfg_idx==5)?0:cfg_idx+1;
                end
                if (strobe && mode_sel) begin
                    buf0[act_idx] <= $signed(ui_in);
                    act_idx <= (act_idx==layer_size[0]-1)?0:act_idx+1;
                end
                if (start) begin
                    st<=S_FETCH; layer_idx<=0; n<=0; i<=0; acc<=0;
                    cur_sel<=0; layer_base<=0;
                    best_idx<=0; best_val<=-19'sd262144; busy_done<=1;
                end
            end else begin
                case (st)
                    S_FETCH: begin
                        // address = layer_base + n*layer_size[layer_idx] + i (byte-addressed)
                        spi_addr <= layer_base + n*layer_size[layer_idx] + i;
                        cur_in   <= cur_sel ? buf1[i] : buf0[i];
                        spi_go<=1; st<=S_WAIT;
                    end
                    S_WAIT:  if (spi_ready) st<=S_MAC;
                    S_MAC: begin
                        acc <= acc + spi_data * cur_in;
                        if (i == layer_size[layer_idx]-1)
                            st<=S_ACT;
                        else begin i<=i+1; st<=S_FETCH; end
                    end
                    S_ACT: begin
                        if (layer_idx != layer_count-1) begin
                            if (cur_sel) buf0[n] <= (act_sel==0)? relu(acc) : acc[7:0];
                            else         buf1[n] <= (act_sel==0)? relu(acc) : acc[7:0];
                        end
                        st<=S_NEXT;
                    end
                    S_NEXT: begin
                        acc<=0; i<=0;
                        if (layer_idx != layer_count-1) begin
                            if (n==layer_size[layer_idx+1]-1) begin
                                n<=0;
                                layer_base <= layer_base + layer_size[layer_idx]*layer_size[layer_idx+1];
                                cur_sel <= ~cur_sel;
                                layer_idx<=layer_idx+1;
                                st<=S_FETCH;
                            end else begin n<=n+1; st<=S_FETCH; end
                        end else begin
                            // final layer: track argmax of raw acc
                            if (acc > best_val) begin best_val<=acc; best_idx<=n; end
                            if (n==layer_size[layer_idx+1]-1) st<=S_ARG;
                            else begin n<=n+1; st<=S_FETCH; end
                        end
                    end
                    S_ARG:  st<=S_DONE;
                    S_DONE: begin busy_done<=0; st<=S_IDLE; end
                    default: st<=S_IDLE;
                endcase
            end
        end
    end

    assign uio_out = uio_out_i;
    assign uo_out  = {4'b0, best_idx};

    // silence unused
    wire _unused = &{ena, uio_in[0], uio_in[1], uio_in[2], uio_in[5], 1'b0};

endmodule

`default_nettype wire
