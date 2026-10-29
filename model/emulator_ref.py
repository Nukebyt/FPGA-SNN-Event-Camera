"""Float reference event emulator (what the integer model and the RTL approximate).

Same transform as the LUT (L = ln(1 + p)), but:
  * contrast threshold in natural-log units,
  * several events per pixel per frame (n = floor(|d| / theta)), reference moves by n * theta,
  * optional linear interpolation of event times inside the frame interval (v2e style),
  * optional leak-free noise events (Poisson, per pixel).

It exists to answer "what do we lose by going to 1 event/pixel/frame + 8-bit log + one timestamp
per frame?" (roadmap 2.3, 2.5, 2.6). Events are returned as float-time arrays, not packed words.
"""
import numpy as np


def emulate_float(frames, fps=30.0, theta_ln=0.2, interpolate=True, noise_rate_hz=0.0, seed=0):
    """Returns dict of arrays t (seconds), x, y, pol (1 = ON), sorted by time."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / fps
    ref = None
    T, X, Y, P = [], [], [], []
    for k, f in enumerate(frames):
        cur = np.log1p(np.asarray(f, dtype=np.float64))
        if ref is None:
            ref = cur.copy()
            continue
        d = cur - ref
        n = np.floor(np.abs(d) / theta_ln).astype(np.int64)
        sign = np.sign(d)
        t0 = (k - 1) * dt
        ys, xs = np.nonzero(n)
        for y, x in zip(ys, xs):
            nn = n[y, x]
            j = np.arange(1, nn + 1)
            frac = (j * theta_ln / abs(d[y, x])) if interpolate else np.ones(nn)
            T.append(t0 + frac * dt)
            X.append(np.full(nn, x))
            Y.append(np.full(nn, y))
            P.append(np.full(nn, 1 if sign[y, x] > 0 else 0))
        ref = ref + sign * n * theta_ln
        if noise_rate_hz > 0:
            m = rng.poisson(noise_rate_hz * dt, size=ref.shape)
            ny, nx = np.nonzero(m)
            if len(ny):
                T.append(t0 + rng.random(len(ny)) * dt)
                X.append(nx)
                Y.append(ny)
                P.append(rng.integers(0, 2, len(ny)))
    if not T:
        z = np.zeros(0)
        return {"t": z, "x": z.astype(int), "y": z.astype(int), "pol": z.astype(int)}
    t = np.concatenate(T)
    order = np.argsort(t, kind="stable")
    return {
        "t": t[order],
        "x": np.concatenate(X)[order].astype(int),
        "y": np.concatenate(Y)[order].astype(int),
        "pol": np.concatenate(P)[order].astype(int),
    }
