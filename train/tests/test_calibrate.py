import torch

from model.snn_config import SNNConfig
from train.snn_qat import QATSNN, calibrate_init


def test_calibrate_init_hits_target_firing():
    torch.manual_seed(0)
    cfg = SNNConfig(in_hw=32, c1=8, c2=16, classes=6)
    x = (torch.rand(6, 20, 2, 32, 32) < 0.12).float()
    std = calibrate_init(cfg, x, targets=(0.08, 0.10))
    assert 1.0 < std[0] < 43 and 1.0 < std[1] < 43 and std[2] > 0
    net = QATSNN(cfg, init_std_int=std)
    with torch.no_grad():
        _, rec = net(x, return_spikes=True)
    f1 = sum(r[0].mean().item() for r in rec) / len(rec)
    f2 = sum(r[1].mean().item() for r in rec) / len(rec)
    assert 0.04 < f1 < 0.16, f1          # random weights differ from the calibration draw, so allow a band
    assert 0.03 < f2 < 0.25, f2
