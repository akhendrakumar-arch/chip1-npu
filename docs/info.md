<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

This is a small, reconfigurable INT8 MLP inference accelerator. Weights are not
stored on-chip: the chip is an SPI master that fetches INT8 weight bytes
on demand from an off-chip source (an RP2040 running spi-ram-emu, or any
device speaking a simple `0x03`-read-command SPI protocol with a 24-bit
address). Activations are streamed in over `ui_in`.

The datapath is a single time-multiplexed MAC unit that is folded over a
small 2-layer MLP (up to 4 inputs, 4 hidden neurons, 4 output classes):
for each neuron it fetches a weight byte over SPI, multiplies it by the
matching activation/hidden value, and accumulates. After the last term
for a neuron it applies ReLU (or passes the raw value through, depending
on configuration) and moves to the next neuron/layer. Once the output
layer is done, it reports the argmax class on `uo_out[3:0]`.

"Reconfigurable" means the same silicon can run different small models:
the number of active inputs, number of output classes, and activation
function are all set at runtime via the config registers, and the
weights themselves live entirely off-chip.

## How to test

1. Hold `rst_n` low for a few clock cycles, then release it.
2. Load configuration (3 bytes, sent with `uio_in[6]`=0 for "config load"
   and a `uio_in[7]` strobe pulse per byte):
   - byte 0: `input_len` (number of active inputs, up to 4)
   - byte 1: `num_out` (number of active output classes, up to 4)
   - byte 2: `act_sel` (0 = ReLU, 1 = identity/passthrough)
3. Load activations: send `input_len` bytes on `ui_in` with `uio_in[6]`=1
   ("data load") and a strobe pulse per byte.
4. Pulse `uio_in[4]` (`start`) for one clock cycle.
5. Poll `uio_in`/`uio_out[5]` (`busy_done`): it goes high while the
   inference runs and drops back to low once it's finished.
6. Read the predicted class on `uo_out[3:0]`.

During all of this, the chip drives the SPI bus (`uio_out[0]`=SCLK,
`uio_out[1]`=CS_N, `uio_out[2]`=MOSI) and reads weight bytes back on
`uio_in[3]` (MISO), issuing a `0x03` read command plus a 24-bit address
for every weight byte it needs.

See `test/test.py` for a complete cocotb example that drives this
sequence end-to-end.

## External hardware

Weights are supplied by an external SPI memory/emulator (e.g. an RP2040
running spi-ram-emu) wired to `uio[0:3]`. No other external hardware is
required; without a weight source connected, `MISO` can be tied to a
fixed level for basic functional testing (as the testbench does).
