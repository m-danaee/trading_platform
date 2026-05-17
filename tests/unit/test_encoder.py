"""
Unit tests for gpu_fuzzy_trader/features/encoder.py

Tests cover:
  - get_dont_care: correct sentinels for all six modes
  - encode_condition: correct fuzzy value names for all modes and all valid genes
  - encode_condition: raises ConfigurationError on dont_care gene
  - encode_condition: raises ConfigurationError on out-of-range gene
  - encode_condition: raises ConfigurationError on unknown mode
  - decode_chromosome: skips dont_care genes, returns correct conditions
  - decode_chromosome: handles all-dont_care chromosome (returns empty list)
  - decode_chromosome: handles no-dont_care chromosome (returns all conditions)
  - decode_chromosome: raises ConfigurationError on length mismatch
  - decode_chromosome: raises ConfigurationError on out-of-range gene
  - Encoder class: delegates correctly to module-level functions
"""

import numpy as np
import pytest

from gpu_fuzzy_trader.features.encoder import (
    ConfigurationError,
    Encoder,
    decode_chromosome,
    encode_condition,
    get_dont_care,
)


# ---------------------------------------------------------------------------
# get_dont_care
# ---------------------------------------------------------------------------

class TestGetDontCare:
    def test_binary(self):
        assert get_dont_care("binary") == 2

    def test_ternary(self):
        assert get_dont_care("ternary") == 3

    def test_positive(self):
        assert get_dont_care("positive") == 5

    def test_sparse_positive(self):
        assert get_dont_care("sparse_positive") == 5

    def test_sparse_signed(self):
        assert get_dont_care("sparse_signed") == 5

    def test_signed(self):
        assert get_dont_care("signed") == 10

    def test_unknown_mode_raises(self):
        with pytest.raises(ConfigurationError):
            get_dont_care("unknown_mode")


# ---------------------------------------------------------------------------
# encode_condition — fuzzy value name mappings
# ---------------------------------------------------------------------------

class TestEncodeConditionBinary:
    def test_gene_0(self):
        assert encode_condition("my_feat", 0, "binary") == "[my_feat] IS Inactive (0)"

    def test_gene_1(self):
        assert encode_condition("my_feat", 1, "binary") == "[my_feat] IS Active (1)"

    def test_dont_care_raises(self):
        with pytest.raises(ConfigurationError):
            encode_condition("my_feat", 2, "binary")

    def test_out_of_range_raises(self):
        with pytest.raises(ConfigurationError):
            encode_condition("my_feat", 3, "binary")


class TestEncodeConditionTernary:
    def test_gene_0(self):
        assert encode_condition("f", 0, "ternary") == "[f] IS Negative (-1)"

    def test_gene_1(self):
        assert encode_condition("f", 1, "ternary") == "[f] IS Neutral (0)"

    def test_gene_2(self):
        assert encode_condition("f", 2, "ternary") == "[f] IS Positive (1)"

    def test_dont_care_raises(self):
        with pytest.raises(ConfigurationError):
            encode_condition("f", 3, "ternary")


class TestEncodeConditionPositive:
    EXPECTED = ["Very Low", "Low", "Medium", "High", "Very High"]

    @pytest.mark.parametrize("gene,name", enumerate(EXPECTED))
    def test_gene(self, gene, name):
        assert encode_condition("feat", gene, "positive") == f"[feat] IS {name}"

    def test_dont_care_raises(self):
        with pytest.raises(ConfigurationError):
            encode_condition("feat", 5, "positive")


class TestEncodeConditionSparsePositive:
    EXPECTED = ["Very Low", "Low", "Medium", "High", "Very High"]

    @pytest.mark.parametrize("gene,name", enumerate(EXPECTED))
    def test_gene(self, gene, name):
        assert encode_condition("feat", gene, "sparse_positive") == f"[feat] IS {name}"

    def test_dont_care_raises(self):
        with pytest.raises(ConfigurationError):
            encode_condition("feat", 5, "sparse_positive")


class TestEncodeConditionSparseSigned:
    EXPECTED = [
        "Strong Negative",
        "Weak Negative",
        "Exactly Zero",
        "Weak Positive",
        "Strong Positive",
    ]

    @pytest.mark.parametrize("gene,name", enumerate(EXPECTED))
    def test_gene(self, gene, name):
        assert encode_condition("feat", gene, "sparse_signed") == f"[feat] IS {name}"

    def test_dont_care_raises(self):
        with pytest.raises(ConfigurationError):
            encode_condition("feat", 5, "sparse_signed")


class TestEncodeConditionSigned:
    EXPECTED = [
        "Extreme Bearish",
        "Strong Bearish",
        "Bearish",
        "Weak Bearish",
        "Neutral Negative",
        "Neutral Positive",
        "Weak Bullish",
        "Bullish",
        "Strong Bullish",
        "Extreme Bullish",
    ]

    @pytest.mark.parametrize("gene,name", enumerate(EXPECTED))
    def test_gene(self, gene, name):
        assert encode_condition("feat", gene, "signed") == f"[feat] IS {name}"

    def test_dont_care_raises(self):
        with pytest.raises(ConfigurationError):
            encode_condition("feat", 10, "signed")

    def test_out_of_range_raises(self):
        with pytest.raises(ConfigurationError):
            encode_condition("feat", 11, "signed")


class TestEncodeConditionErrors:
    def test_unknown_mode_raises(self):
        with pytest.raises(ConfigurationError):
            encode_condition("feat", 0, "nonexistent")

    def test_negative_gene_raises(self):
        with pytest.raises(ConfigurationError):
            encode_condition("feat", -1, "positive")

    def test_condition_format_has_brackets(self):
        result = encode_condition("dmi_balance_14", 2, "signed")
        assert result.startswith("[dmi_balance_14]")
        assert " IS " in result

    def test_condition_format_exact(self):
        result = encode_condition("vol_ratio_20_100", 0, "positive")
        assert result == "[vol_ratio_20_100] IS Very Low"


# ---------------------------------------------------------------------------
# decode_chromosome
# ---------------------------------------------------------------------------

class TestDecodeChromosome:
    def _make_infos(self, names_modes):
        return [{"name": n, "mode": m} for n, m in names_modes]

    def test_all_dont_care_returns_empty(self):
        infos = self._make_infos([("a", "binary"), ("b", "ternary"), ("c", "positive")])
        chromosome = np.array([2, 3, 5])  # all dont_care
        result = decode_chromosome(chromosome, infos)
        assert result == []

    def test_no_dont_care_returns_all_conditions(self):
        infos = self._make_infos([("a", "binary"), ("b", "ternary")])
        chromosome = np.array([0, 1])
        result = decode_chromosome(chromosome, infos)
        assert result == ["[a] IS Inactive (0)", "[b] IS Neutral (0)"]

    def test_mixed_skips_dont_care(self):
        infos = self._make_infos([
            ("feat_a", "positive"),
            ("feat_b", "signed"),
            ("feat_c", "binary"),
        ])
        # feat_b is dont_care (10), feat_a and feat_c are active
        chromosome = np.array([3, 10, 1])
        result = decode_chromosome(chromosome, infos)
        assert result == [
            "[feat_a] IS High",
            "[feat_c] IS Active (1)",
        ]

    def test_order_preserved(self):
        infos = self._make_infos([
            ("first", "binary"),
            ("second", "ternary"),
            ("third", "sparse_signed"),
        ])
        chromosome = np.array([1, 2, 0])
        result = decode_chromosome(chromosome, infos)
        assert result[0] == "[first] IS Active (1)"
        assert result[1] == "[second] IS Positive (1)"
        assert result[2] == "[third] IS Strong Negative"

    def test_length_mismatch_raises(self):
        infos = self._make_infos([("a", "binary"), ("b", "ternary")])
        chromosome = np.array([0])  # too short
        with pytest.raises(ConfigurationError):
            decode_chromosome(chromosome, infos)

    def test_out_of_range_gene_raises(self):
        infos = self._make_infos([("a", "binary")])
        chromosome = np.array([5])  # out of range for binary (valid: 0, 1; dont_care: 2)
        with pytest.raises(ConfigurationError):
            decode_chromosome(chromosome, infos)

    def test_unknown_mode_raises(self):
        infos = [{"name": "a", "mode": "bad_mode"}]
        chromosome = np.array([0])
        with pytest.raises(ConfigurationError):
            decode_chromosome(chromosome, infos)

    def test_single_active_condition(self):
        infos = self._make_infos([("rsi", "positive")])
        chromosome = np.array([4])
        result = decode_chromosome(chromosome, infos)
        assert result == ["[rsi] IS Very High"]

    def test_sparse_positive_dont_care_skipped(self):
        infos = self._make_infos([("a", "sparse_positive"), ("b", "sparse_positive")])
        chromosome = np.array([5, 2])  # first is dont_care
        result = decode_chromosome(chromosome, infos)
        assert result == ["[b] IS Medium"]

    def test_signed_all_values(self):
        """All 10 signed values decode correctly via decode_chromosome."""
        signed_names = [
            "Extreme Bearish", "Strong Bearish", "Bearish", "Weak Bearish",
            "Neutral Negative", "Neutral Positive", "Weak Bullish", "Bullish",
            "Strong Bullish", "Extreme Bullish",
        ]
        for gene, expected_name in enumerate(signed_names):
            infos = [{"name": "macd", "mode": "signed"}]
            chromosome = np.array([gene])
            result = decode_chromosome(chromosome, infos)
            assert result == [f"[macd] IS {expected_name}"]

    def test_numpy_int_types_accepted(self):
        """Chromosome with numpy int32/int64 values should work."""
        infos = self._make_infos([("a", "binary")])
        chromosome = np.array([1], dtype=np.int32)
        result = decode_chromosome(chromosome, infos)
        assert result == ["[a] IS Active (1)"]


# ---------------------------------------------------------------------------
# Encoder class (OOP wrapper)
# ---------------------------------------------------------------------------

class TestEncoderClass:
    def test_get_dont_care_delegates(self):
        enc = Encoder()
        assert enc.get_dont_care("binary") == 2
        assert enc.get_dont_care("signed") == 10

    def test_encode_condition_delegates(self):
        enc = Encoder()
        assert enc.encode_condition("feat", 0, "ternary") == "[feat] IS Negative (-1)"

    def test_decode_chromosome_delegates(self):
        enc = Encoder()
        infos = [{"name": "x", "mode": "binary"}, {"name": "y", "mode": "binary"}]
        chromosome = np.array([0, 2])  # second is dont_care
        result = enc.decode_chromosome(chromosome, infos)
        assert result == ["[x] IS Inactive (0)"]

    def test_static_methods_callable_on_class(self):
        """Static methods should be callable on the class itself."""
        assert Encoder.get_dont_care("ternary") == 3
        assert Encoder.encode_condition("f", 1, "binary") == "[f] IS Active (1)"
