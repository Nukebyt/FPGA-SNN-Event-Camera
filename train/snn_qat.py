"""Quantization-aware SNN whose forward pass IS the integer arithmetic of model/snn_int.py.

Every tensor in the forward pass holds exact integers (float32 is exact below 2**24):
  * weights: round + clamp to int8 on the fly, straight-through gradient to the latent float weights
  * leak:    v - floor(v / 2**k)        (== v - (v >> k))
  * spikes:  heaviside(v - thr) forward, fast-sigmoid surrogate backward (non-zero tails so resting neurons still learn)
  * reset:   v - thr * spike (the spike is detached in the reset path)
so `QATSNN.forward` and `IntSNN.run` must agree exactly (train/tests/test_qat_vs_int.py).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.snn_config import SNNConfig


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, thr):
        ctx.save_for_backward(v)
        ctx.thr = thr
        return (v >= thr).to(v.dtype)

    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        # fast-sigmoid shape with non-zero tails; slope scaled by 1/thr because membranes are in integer units
        # (without it the backward signal grows by ~|w| per layer and per step and the gradient explodes)
        sg = (1.0 / ctx.thr) / (1.0 + (v - ctx.thr).abs() / (0.5 * ctx.thr)) ** 2
        return g * sg, None


def lif_step(v, current, cfg, thr):
    v = v - torch.floor(v / (1 << cfg.leak_shift))
    v = torch.clamp(v + current, cfg.vmin, cfg.vmax)
    s = SpikeFn.apply(v, float(thr))
    v = v - thr * s.detach()
    return v, s


def or_pool(s):
    return F.max_pool2d(s, 2)


class QATSNN(nn.Module):
    def __init__(self, cfg=SNNConfig(), init_std_int=None, quantize=True):
        """init_std_int: initial int-weight std per conv layer, then fc (default 40, 20, [10,] 6)."""
        super().__init__()
        self.cfg = cfg
        self.quantize = quantize          # False = float-weight SNN, same neuron (roadmap 3.3 comparison)
        k = cfg.wscale
        if init_std_int is None:
            init_std_int = [40.0, 20.0] + [10.0] * (cfg.n_conv - 2) + [6.0]
        assert len(init_std_int) == cfg.n_conv + 1
        chans = [cfg.in_ch] + cfg.channels
        self.wc = nn.ParameterList([nn.Parameter(torch.randn(chans[i + 1], chans[i], 3, 3) * init_std_int[i] / k)
                                    for i in range(cfg.n_conv)])
        self.w_fc = nn.Parameter(torch.randn(cfg.classes, cfg.fc_in) * init_std_int[-1] / k)

    def quant(self, w):
        if not self.quantize:
            return w * self.cfg.wscale
        wq = torch.clamp(torch.round(w * self.cfg.wscale), -128, 127)
        return w * self.cfg.wscale + (wq - w * self.cfg.wscale).detach()     # STE, value == wq exactly

    def int_weights(self):
        """int32 numpy arrays: one per conv layer, then fc."""
        with torch.no_grad():
            return [torch.clamp(torch.round(w * self.cfg.wscale), -128, 127).to(torch.int32).cpu().numpy()
                    for w in list(self.wc) + [self.w_fc]]

    def forward(self, x, return_spikes=False):
        """x [B, T, 2, H, H] in {0,1}. Returns class scores [B, classes] (float, integer valued while |score| < 2**24)."""
        c = self.cfg
        B, T = x.shape[:2]
        wq = [self.quant(w) for w in self.wc]
        w_fc = self.quant(self.w_fc)
        v = [x.new_zeros(B, ch, c.in_hw >> i, c.in_hw >> i) for i, ch in enumerate(c.channels)]
        v_out = x.new_zeros(B, c.classes)
        counts = x.new_zeros(B, c.classes)
        rec = []
        for t in range(T):
            cur, spikes = x[:, t], []
            for i, w in enumerate(wq):
                v[i], s = lif_step(v[i], F.conv2d(cur, w, padding=1), c, c.thr[i])
                spikes.append(s)
                cur = or_pool(s)
            i_out = F.linear(cur.flatten(1), w_fc)
            if c.readout == "sum":
                out = i_out                                  # class currents, accumulated below
            else:
                v_out, out = lif_step(v_out, i_out, c, c.thr[-1])
            counts = counts + out
            if return_spikes:
                rec.append(tuple(s.detach() for s in spikes) + (out.detach(),))
        return (counts, rec) if return_spikes else counts


class ANNBaseline(nn.Module):
    """Same topology on accumulated event frames (roadmap 3.4): ReLU, max-pool, no biases."""

    def __init__(self, cfg=SNNConfig()):
        super().__init__()
        chans = [cfg.in_ch] + cfg.channels
        self.convs = nn.ModuleList([nn.Conv2d(chans[i], chans[i + 1], 3, padding=1, bias=False) for i in range(cfg.n_conv)])
        self.fc = nn.Linear(cfg.fc_in, cfg.classes, bias=False)

    def forward(self, x):                      # x [B, T, 2, H, W] -> mean over time
        a = x.mean(1)
        for conv in self.convs:
            a = F.max_pool2d(F.relu(conv(a)), 2)
        return self.fc(a.flatten(1))


def calibrate_init(cfg, x, targets=(0.08, 0.10), fc_std_at_1024=3.0, iters=14):
    """Pick initial int-weight std per conv layer so each fires at about `targets` per neuron per step.

    Bisects layer 1's std on the input batch x [B, T, 2, H, H], then each later layer on the previous layer's
    spikes (targets[0] for layer 1, targets[1] for the rest). The FC std (sum readout: only its scale relative to
    the logit scale matters) is scaled as 1/sqrt(fan_in) from `fc_std_at_1024` at 1024 inputs.
    Returns one std per conv layer, then the fc std.
    """
    import math
    dev, c = x.device, cfg
    chans = [c.in_ch] + c.channels

    def firing(std, layer, s_in):                  # layer is 0-based
        torch.manual_seed(1234)
        w = torch.clamp(torch.round(torch.randn(chans[layer + 1], chans[layer], 3, 3, device=dev) * std), -128, 127)
        hw = c.in_hw >> layer
        v = torch.zeros(x.shape[0], chans[layer + 1], hw, hw, device=dev)
        src = (lambda t: x[:, t]) if layer == 0 else (lambda t: or_pool(s_in[t]))
        rate, outs = 0.0, []
        with torch.no_grad():
            for t in range(x.shape[1]):
                v, s = lif_step(v, F.conv2d(src(t), w, padding=1), c, c.thr[layer])
                rate += s.mean().item()
                outs.append(s)
        return rate / x.shape[1], outs

    stds, s_prev = [], None
    for layer in range(c.n_conv):
        target = targets[0] if layer == 0 else targets[1]
        lo, hi = 1.0, 127.0 / 3
        for _ in range(iters):                      # log-scale bisection, firing is increasing in std
            mid = math.sqrt(lo * hi)
            r, _ = firing(mid, layer, s_prev)
            lo, hi = (mid, hi) if r < target else (lo, mid)
        std = math.sqrt(lo * hi)
        _, s_prev = firing(std, layer, s_prev)
        stds.append(std)
    stds.append(fc_std_at_1024 * math.sqrt(1024.0 / cfg.fc_in))
    return tuple(stds)


def export_int(model, out_dir):
    """Writes snn_config.json + weights_int8.npz (consumed by IntSNN.load) into out_dir."""
    from pathlib import Path
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    ws = model.int_weights()
    (d / "snn_config.json").write_text(model.cfg.to_json())
    arrays = {"conv%d" % (i + 1): w.astype(np.int8) for i, w in enumerate(ws[:-1])}
    np.savez(d / "weights_int8.npz", fc=ws[-1].astype(np.int8), **arrays)
