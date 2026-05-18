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
