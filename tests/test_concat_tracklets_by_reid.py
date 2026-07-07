"""Tests for concat_tracklets_by_reid embedding handling."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sn_gamestate.concat_tracklets_by_reid.concat_tracklets_by_reid_api import ConcatTrackletsByReid
from sn_gamestate.reid.embedding_utils import mean_track_embedding


def test_mean_track_embedding_ignores_nan_rows():
    good = np.random.randn(256)
    assert mean_track_embedding([np.nan, good, None]) is not None
    out = mean_track_embedding([np.nan, good, None])
    np.testing.assert_allclose(out, good, rtol=1e-5)


def test_concat_by_reid_skips_nan_only_tracks():
    emb = np.random.randn(1, 256)
    det = pd.DataFrame({
        "track_id": [1, 1, 2, 2],
        "image_id": ["a", "b", "c", "d"],
        "embeddings": [emb[0], np.nan, np.nan, np.nan],
    })
    out = ConcatTrackletsByReid(threshold=0.1).process(det, pd.DataFrame())
    assert set(out.track_id) == {1, 2}
