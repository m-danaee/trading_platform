"""Tests for optional memory logging helpers."""

from __future__ import annotations

import os

from gpu_fuzzy_trader._memory import log_memory_rss, release_phase2_resources


def test_log_memory_rss_noop_without_env(monkeypatch):
    monkeypatch.delenv("LOG_MEMORY", raising=False)
    log_memory_rss("test")  # should not raise


def test_log_memory_rss_with_env(monkeypatch):
    monkeypatch.setenv("LOG_MEMORY", "1")
    try:
        import psutil  # noqa: F401
    except ImportError:
        log_memory_rss("test")
        return
    log_memory_rss("test")


def test_release_phase2_resources():
    release_phase2_resources()


def test_park_engines_clears_engine_handles(monkeypatch) -> None:
    from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator

    gen = object.__new__(Rule_Pool_Generator)
    gen.direction = "long"
    gen._engine = object()
    gen._val_engine = object()
    gen.park_engines()
    assert gen._engine is None
    assert gen._val_engine is None
