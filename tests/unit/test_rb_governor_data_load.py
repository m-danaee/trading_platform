"""Tests for RB Governor internal train/validation data loading."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.rb_governor import _load_internal_split_frames_for_rb
from tests.unit.test_data_splitter import _make_df, _patch_split_paths
from gpu_fuzzy_trader.data.splitter import Data_Splitter


class TestRbGovernorDataLoad:
    def test_load_internal_split_frames_uses_train_70_cache(
        self, tmp_path, monkeypatch,
    ):
        csv_path = tmp_path / "train_2.csv"
        csv_path.write_text("x\n1\n", encoding="utf-8")

        with _patch_split_paths(str(tmp_path))() as paths:
            df = _make_df({1: 2000})
            Data_Splitter().split_and_persist(df)
            os.utime(csv_path, (1, 1))
            for path in paths.values():
                os.utime(path, (2, 2))

        monkeypatch.setattr(_cfg, "TRAIN_CSV_PATH", str(csv_path))
        monkeypatch.setattr(_cfg, "TRAIN_70_PATH", paths["train"])
        monkeypatch.setattr(_cfg, "VALIDATION_30_PATH", paths["val"])
        monkeypatch.setattr(_cfg, "VALIDATION_FITNESS_PATH", paths["fitness"])
        monkeypatch.setattr(
            _cfg, "VALIDATION_SELECTION_PATH", paths["selection"])
        monkeypatch.setattr(_cfg, "CV_FOLDS_MANIFEST_PATH", paths["manifest"])
        monkeypatch.setattr(_cfg, "SPLIT_MODE", "holdout")

        train_df, val_df = _load_internal_split_frames_for_rb()
        assert len(train_df) > 0
        assert len(val_df) > 0
