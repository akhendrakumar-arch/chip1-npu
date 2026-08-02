/*
 * tt_um_akhendrakumar_npu
 * Reconfigurable INT8 MLP inference accelerator (weight-streaming).
 *
 * Weights live off-chip in the RP2040 (spi-ram-emu). This chip is the
 * SPI master: it issues a read command + 24-bit address and shifts in
 * INT8 weight bytes. Activations are streamed in over ui_in. The chip
 * does INT8 x INT8 -> INT32 MACs, ReLU, and returns an argmax class.
 *
 * The datapath is time-multiplexed: K parallel MACs are folded over
 * neurons and layers, so the design fits a single Tiny Tapeout tile.
 * "Reconfigurable" = same silicon, different weights + config.
 *
 * NOTE: This is a compact, synthesizable skeleton sized to 1 tile.
 * It exercises the full control path (SPI fetch -> MAC -> ReLU ->
 * argmax) on a small internal shape so it hardens + verifies cleanly.
 * Scale INPUT_N / HIDDEN_N / OUT_N up only if tile area allows.
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

    // ---- small internal shape (fits 1 tile) ----
    localparam INPUT_N  = 4;   // input features (masked via input_len)
    localparam HIDDEN_N = 4;   // hidden neurons
    localparam OUT_N    = 4;   // output classes

    // ---- config registers (loaded when mode_sel=0, on strobe) ----
    reg [7:0] input_len;       // active input length
    reg [7:0] num_out;         // active output classes
    reg [7:0] act_sel;         // 0 = ReLU, 1 = identity
    reg [7:0] cfg_idx;

    // ---- activation buffer (loaded when mode_sel=1, on strobe) ----
    reg signed [7:0] act [0:INPUT_N-1];
    reg [3:0] act_idx;

    // ---- SPI master: read one weight byte by address ----
    // Simplified spi-ram-emu style: assert cs, shift 8-bit read cmd +
    // 24-bit addr, then shift 8 data bits in on miso. One byte per call.
    reg  [2:0]  spi_state;
    reg  [5:0]  spi_bitcnt;
    reg  [12:0] spi_addr;       // low 13 bits of the 24-bit address; bits above are always 0
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
                SPI_ADDR: begin // shift out 24-bit address (top 11 bits are always 0)
                    spi_mosi <= (spi_bitcnt > 13) ? 1'b0 : spi_addr[spi_bitcnt-1];
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
    reg  [3:0]  n;              // current neuron / class
    reg  [3:0]  i;              // current input index
    reg  signed [17:0] acc;      // max |sum of 4 int8*int8 terms| = 4*16384 = 65536, fits in 18 bits
    reg  signed [7:0]  hidden [0:HIDDEN_N-1];
    reg  [3:0]  best_idx;
    reg  signed [17:0] best_val;
    reg         layer;          // 0 = in->hidden, 1 = hidden->out
    localparam S_IDLE=0,S_FETCH=1,S_WAIT=2,S_MAC=3,S_ACT=4,S_NEXT=5,
               S_ARG=6,S_DONE=7;

    function signed [7:0] relu; input signed [17:0] v;
        relu = (v[17]) ? 8'sd0 :
               (v > 18'sd127) ? 8'sd127 : v[7:0];
    endfunction

    reg signed [7:0] cur_in;    // current operand from prev layer / input

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st<=S_IDLE; n<=0; i<=0; acc<=0; layer<=0; spi_go<=0;
            busy_done<=0; best_idx<=0; best_val<=0; spi_addr<=0;
            input_len<=INPUT_N; num_out<=OUT_N; act_sel<=0;
            cfg_idx<=0; act_idx<=0;
        end else begin
            spi_go<=0;
            // ---- config / activation loading (only when idle) ----
            if (st==S_IDLE) begin
                busy_done<=0;
                if (strobe && !mode_sel) begin
                    case (cfg_idx)
                        0: input_len<=ui_in;
                        1: num_out  <=ui_in;
                        2: act_sel  <=ui_in;
                    endcase
                    cfg_idx <= (cfg_idx==2)?0:cfg_idx+1;
                end
                if (strobe && mode_sel) begin
                    act[act_idx] <= $signed(ui_in);
                    act_idx <= (act_idx==INPUT_N-1)?0:act_idx+1;
                end
                if (start) begin
                    st<=S_FETCH; n<=0; i<=0; acc<=0; layer<=0;
                    best_idx<=0; best_val<=-18'sd131072; busy_done<=1;
                end
            end else begin
                case (st)
                    S_FETCH: begin
                        // address = base(layer) + n*len + i  (byte-addressed)
                        spi_addr <= (layer? 13'h1000 : 13'h0000)
                                    + n*input_len + i;
                        cur_in   <= (layer? hidden[i] : act[i]);
                        spi_go<=1; st<=S_WAIT;
                    end
                    S_WAIT:  if (spi_ready) st<=S_MAC;
                    S_MAC: begin
                        acc <= acc + spi_data * cur_in;
                        if (i == (layer? HIDDEN_N-1 : input_len-1))
                            st<=S_ACT;
                        else begin i<=i+1; st<=S_FETCH; end
                    end
                    S_ACT: begin
                        if (layer==0)
                            hidden[n] <= (act_sel==0)? relu(acc) : acc[7:0];
                        st<=S_NEXT;
                    end
                    S_NEXT: begin
                        acc<=0; i<=0;
                        if (layer==0) begin
                            if (n==HIDDEN_N-1) begin n<=0; layer<=1; st<=S_FETCH; end
                            else begin n<=n+1; st<=S_FETCH; end
                        end else begin
                            // output layer: track argmax of raw acc
                            if (acc > best_val) begin best_val<=acc; best_idx<=n; end
                            if (n==num_out-1) st<=S_ARG;
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

