import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def strobe(dut, value, mode):
    """Push one byte over ui_in with strobe; mode=0 config, 1 activation."""
    dut.ui_in.value = value
    dut.uio_in.value = (mode << 6) | (1 << 7)   # mode_sel, strobe=1
    await RisingEdge(dut.clk)
    dut.uio_in.value = (mode << 6)              # strobe=0
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_npu_runs(dut):
    """Reset, load config + activations, run one inference to completion."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # reset
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Emulate the RP2040 SPI weight source: hold miso (uio_in[3]) high so
    # each fetched weight byte is deterministic. Real board uses spi-ram-emu.
    # We OR miso into every uio_in write via a small helper below.

    async def strobe_miso(value, mode):
        dut.ui_in.value = value
        dut.uio_in.value = (1 << 3) | (mode << 6) | (1 << 7)
        await RisingEdge(dut.clk)
        dut.uio_in.value = (1 << 3) | (mode << 6)
        await RisingEdge(dut.clk)

    # config: input_len=4, num_out=4, act_sel=0 (ReLU)
    await strobe_miso(4, 0)
    await strobe_miso(4, 0)
    await strobe_miso(0, 0)

    # activations = 1,2,3,4
    for a in (1, 2, 3, 4):
        await strobe_miso(a, 1)

    # pulse start (uio_in[4]), keep miso high
    dut.uio_in.value = (1 << 3) | (1 << 4)
    await RisingEdge(dut.clk)
    dut.uio_in.value = (1 << 3)

    # wait until busy_done (uio_out[5]) asserts then deasserts
    for _ in range(200):
        await RisingEdge(dut.clk)
        if (int(dut.uio_out.value) >> 5) & 1:
            break

    finished = False
    for _ in range(200000):
        await RisingEdge(dut.clk)
        if ((int(dut.uio_out.value) >> 5) & 1) == 0:
            finished = True
            break

    result = int(dut.uo_out.value) & 0xF
    dut._log.info(f"inference finished={finished}, class={result}")
    assert finished, "FSM did not complete (busy_done stuck high)"
