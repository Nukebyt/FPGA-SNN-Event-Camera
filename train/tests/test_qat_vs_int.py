"""Roadmap 3.7: the QAT forward pass and the numpy integer model agree bit for bit."""
import numpy as np
import torch

from model.snn_config import SNNConfig
from model.snn_int import IntSNN
from train.snn_qat import QATSNN


def _check(cfg, density, seed, T=12, hw=32):
    torch.manual_seed(seed)
    _check_net(QATSNN(cfg, init_std_int=(60.0, 40.0, 14.0)), cfg, density, seed, T, hw)


def _check_net(net, cfg, density, seed, T=12, hw=32):
    rng = np.random.default_rng(seed)
    x = (rng.random((3, T, 2, hw, hw)) < density).astype(np.float32)
    with torch.no_grad():
        counts, rec = net(torch.from_numpy(x), return_spikes=True)
    ref = IntSNN(cfg, *net.int_weights())
    total = 0
    for b in range(x.shape[0]):
        c, r = ref.run(x[b].astype(np.int32), record=True)
        assert np.array_equal(c, counts[b].numpy().astype(np.int64)), (b, c, counts[b])
        for t, step in enumerate(r):
            assert len(step) == len(rec[t])
            for k, s_int in enumerate(step):
                ref_dtype = np.int64 if k == len(step) - 1 else np.int32
                assert np.array_equal(s_int, rec[t][k][b].numpy().astype(ref_dtype)), (b, t, k)
        total += int(np.abs(c).sum()) + sum(int(s.sum()) for step in r for s in step[:-1])
    assert total > 0, "test is vacuous: nothing spiked"


def test_qat_matches_int_model_default():
    _check(SNNConfig(classes=11), density=0.08, seed=0)               # readout "sum"


def test_qat_matches_int_model_spike_readout():
    _check(SNNConfig(classes=11, readout="spike"), density=0.08, seed=3)


def test_qat_matches_int_model_other_neuron_params():
    _check(SNNConfig(classes=6, leak_shift=3, thr=[48, 80, 40], vbits=10, readout="spike"), density=0.15, seed=1)


def test_saturation_path_is_exercised():
    cfg = SNNConfig(classes=6, vbits=8, thr=[100, 100, 100])
    _check(cfg, density=0.3, seed=2)


def test_qat_matches_int_model_64x64_both_readouts():
    for readout, seed in (("sum", 5), ("spike", 6)):
        cfg = SNNConfig(in_hw=64, c1=6, c2=10, classes=11, readout=readout)
        assert cfg.down == 2 and cfg.fc_in == 10 * 16 * 16
        _check(cfg, density=0.05, seed=seed, T=8, hw=64)


def test_qat_matches_int_model_third_conv_layer():
    for readout, seed, hw in (("sum", 7, 32), ("spike", 8, 32), ("sum", 9, 64)):
        cfg = SNNConfig(in_hw=hw, c1=6, c2=10, c3=12, classes=11, readout=readout)
        assert cfg.n_conv == 3 and len(cfg.thr) == 4 and cfg.fc_in == 12 * (hw // 8) ** 2
        net = QATSNN(cfg, init_std_int=(60.0, 40.0, 30.0, 14.0))
        assert len(net.int_weights()) == 4
        _check_net(net, cfg, density=0.08 if hw == 32 else 0.04, seed=seed, T=8, hw=hw)
