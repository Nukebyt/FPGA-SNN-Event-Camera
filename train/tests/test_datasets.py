import numpy as np

from train.datasets import WindowDataset, load_cache, pack, save_cache, unpack


def _samples():
    rng = np.random.default_rng(0)
    return [(rng.random((n, 2, 32, 32)) < 0.05).astype(np.uint8) for n in (10, 25, 7)], [0, 1, 2]


def test_pack_roundtrip():
    s, _ = _samples()
    for a in s:
        assert np.array_equal(unpack(pack(a)), a)


def test_cache_roundtrip(tmp_path):
    s, y = _samples()
    save_cache(tmp_path / "c.npz", s, y)
    s2, y2 = load_cache(tmp_path / "c.npz")
    assert list(y2) == y and all(np.array_equal(a, b) for a, b in zip(s, s2))


def test_eval_windows_cover_every_bin_once_and_pad():
    s, y = _samples()
    ds = WindowDataset(s, y, T=8, train=False)
    assert len(ds) == 2 + 4 + 1
    total = 0
    for i in range(len(ds)):
        w, lab, sid = ds[i]
        assert w.shape == (8, 2, 32, 32)
        total += int(w.sum())
    assert total == sum(int(a.sum()) for a in s)


def test_train_window_shape_and_short_sample_padding():
    s, y = _samples()
    ds = WindowDataset(s, y, T=16, train=True, shift=0)
    for i in range(3):
        w, lab, sid = ds[i]
        assert w.shape == (16, 2, 32, 32) and lab == y[i]


def test_64x64_cache_roundtrip_and_windows(tmp_path):
    rng = np.random.default_rng(1)
    s = [(rng.random((n, 2, 64, 64)) < 0.05).astype(np.uint8) for n in (12, 20)]
    save_cache(tmp_path / "c64.npz", s, [3, 4])
    s2, y2 = load_cache(tmp_path / "c64.npz")
    assert s2[0].shape == (12, 2, 64, 64) and all(np.array_equal(a, b) for a, b in zip(s, s2))
    ds = WindowDataset(s2, y2, T=8, train=True)
    assert ds.hw == 64 and ds.shift == 4
    w, lab, _ = ds[0]
    assert w.shape == (8, 2, 64, 64)
    ev = WindowDataset(s2, y2, T=8, train=False)
    assert sum(int(ev[i][0].sum()) for i in range(len(ev))) == sum(int(a.sum()) for a in s)


def test_augmentations_flip_drop_speed():
    from train.datasets import FLIP_MAP_DVS
    rng = np.random.default_rng(2)
    base = (rng.random((40, 2, 32, 32)) < 0.2).astype(np.uint8)
    # flip: with prob 1 the window is mirrored and the label remapped (use many draws, flip happens ~50%)
    ds = WindowDataset([base], [1], T=8, train=True, shift=0, flip_map=FLIP_MAP_DVS, seed=0)
    labels = {ds[0][1] for _ in range(40)}
    assert labels == {1, 2}
    assert FLIP_MAP_DVS[FLIP_MAP_DVS[3]] == 3 and FLIP_MAP_DVS[FLIP_MAP_DVS[4]] == 4      # involution
    # drop: about the requested fraction of active cells is removed
    ds = WindowDataset([base], [0], T=40, train=True, shift=0, drop=0.5, seed=1)
    kept = float(ds[0][0].sum()) / float(base.sum())
    assert 0.4 < kept < 0.6, kept
    # speed: 2x window OR-merged pairs -> never more active cells than the two source bins together
    ds = WindowDataset([base], [0], T=10, train=True, shift=0, speed=1.0, seed=2)
    w = ds[0][0].numpy()
    assert w.shape == (10, 2, 32, 32) and w.max() == 1.0
    assert float(w.mean()) > float(base.mean())          # OR of two bins is denser than a single bin
