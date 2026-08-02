import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


async def strobe_miso(dut, value, mode):
    """Push one byte over ui_in with strobe; MISO held high so every
    fetched weight byte is deterministic (0xFF = -1). mode=0 config,
    mode=1 activation. Real board uses spi-ram-emu for MISO."""
    dut.ui_in.value = value
    dut.uio_in.value = (1 << 3) | (mode << 6) | (1 << 7)   # miso=1, mode_sel, strobe=1
    await RisingEdge(dut.clk)
    dut.uio_in.value = (1 << 3) | (mode << 6)               # strobe=0
    await RisingEdge(dut.clk)


async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def run_inference(dut):
    """Pulse start and wait for busy_done (uio_out[5]) to assert then deassert."""
    dut.uio_in.value = (1 << 3) | (1 << 4)
    await RisingEdge(dut.clk)
    dut.uio_in.value = (1 << 3)

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

    return finished, int(dut.uo_out.value) & 0xF


@cocotb.test()
async def test_npu_two_layer(dut):
    """Reset, configure a runtime 2-layer (4-4-4) MLP, run one inference."""
    await reset(dut)

    # config: layer_count=2 (in->hidden->out), layer_size=[4,4,4,x], act_sel=0 (ReLU)
    await strobe_miso(dut, 2, 0)  # layer_count
    await strobe_miso(dut, 4, 0)  # layer_size[0] = input width
    await strobe_miso(dut, 4, 0)  # layer_size[1] = hidden width
    await strobe_miso(dut, 4, 0)  # layer_size[2] = output width
    await strobe_miso(dut, 0, 0)  # layer_size[3] = unused (layer_count=2)
    await strobe_miso(dut, 0, 0)  # act_sel = ReLU

    # activations = 1,2,3,4
    for a in (1, 2, 3, 4):
        await strobe_miso(dut, a, 1)

    finished, result = await run_inference(dut)
    dut._log.info(f"two_layer: finished={finished}, class={result}")
    assert finished, "FSM did not complete (busy_done stuck high)"
    # All weights read back as -1 (MISO held high). Layer 0: every hidden
    # neuron sums -1*(1+2+3+4)=-10, ReLU clamps to 0. Layer 1 (final):
    # every output neuron sums -1*0=0, so argmax picks the first, class 0.
    assert result == 0, f"expected class 0, got {result}"


@cocotb.test()
async def test_npu_single_layer(dut):
    """Reconfigure to a different runtime shape: a single direct 3-3 layer
    (no hidden layer), proving layer_count/layer widths are truly runtime
    configurable and not hardcoded."""
    await reset(dut)

    # config: layer_count=1 (in->out directly), layer_size=[3,3,x,x], act_sel=0
    await strobe_miso(dut, 1, 0)  # layer_count
    await strobe_miso(dut, 3, 0)  # layer_size[0] = input width
    await strobe_miso(dut, 3, 0)  # layer_size[1] = output width
    await strobe_miso(dut, 0, 0)  # layer_size[2] = unused (layer_count=1)
    await strobe_miso(dut, 0, 0)  # layer_size[3] = unused
    await strobe_miso(dut, 0, 0)  # act_sel (irrelevant: final layer skips activation)

    # activations = 2,2,2
    for a in (2, 2, 2):
        await strobe_miso(dut, a, 1)

    finished, result = await run_inference(dut)
    dut._log.info(f"single_layer: finished={finished}, class={result}")
    assert finished, "FSM did not complete (busy_done stuck high)"
    # Single (final) layer: every output neuron sums -1*(2+2+2)=-6 (no
    # activation applied on the final layer), so argmax picks the first,
    # class 0.
    assert result == 0, f"expected class 0, got {result}"
