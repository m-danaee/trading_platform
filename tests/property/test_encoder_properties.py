"""
Property-based tests for gpu_fuzzy_trader.features.encoder.Encoder

**Validates: Requirements 4.1, 4.2, 4.3**

Property 8: Fuzzy Value Name Encoding Round-Trip
  For any valid (mode, gene) pair where gene != dont_care:
    - encode_condition(feature_name, gene, mode) must return a string that:
        1. Starts with '[feature_name]'
        2. Contains ' IS ' in the middle
        3. Ends with a valid fuzzy value name for that mode
        4. Has the exact format '[feature_name] IS Fuzzy Value Name'

  For any valid chromosome (array of genes):
    - decode_chromosome(chromosome, feature_infos) must:
        1. Return a list of strings
        2. Each string must match the '[feature_name] IS Fuzzy Value Name' pattern
        3. The number of conditions must equal the number of non-dont_care genes

Property 9: Don't-Care Sentinel Correctness
  For any mode, the don't-care sentinel returned by get_dont_care(mode) must:
    1. Equal num_classes for that mode:
         binary → 2, ternary → 3,
         positive / sparse_positive / sparse_signed → 5,
         signed → 10
    2. Cause encode_condition(feature_name, dont_care, mode) to raise
       ConfigurationError for any feature name.
    3. Be strictly greater than all valid gene values for that mode
       (i.e., dont_care > max_valid_gene).
    4. Cause decode_chromosome to skip any gene equal to dont_care —
       the gene must not appear in the output condition list.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from gpu_fuzzy_trader.features.encoder import (
    ConfigurationError,
    Encoder,
    decode_chromosome,
    encode_condition,
    get_dont_care,
)


# ---------------------------------------------------------------------------
# Constants — expected sentinels, num_classes, and fuzzy value names per mode
# ---------------------------------------------------------------------------

# Maps mode → (num_classes, dont_care_sentinel)
_MODE_SPECS: dict[str, tuple[int, int]] = {
    "binary":          (2,  2),
    "ternary":         (3,  3),
    "positive":        (5,  5),
    "sparse_positive": (5,  5),
    "sparse_signed":   (5,  5),
    "signed":          (10, 10),
}

_ALL_MODES = list(_MODE_SPECS.keys())

# Maps mode → list of fuzzy value names (index = gene value)
_FUZZY_VALUE_NAMES: dict[str, list[str]] = {
    "binary": [
        "Inactive (0)",
        "Active (1)",
    ],
    "ternary": [
        "Negative (-1)",
        "Neutral (0)",
        "Positive (1)",
    ],
    "positive": [
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High",
    ],
    "sparse_positive": [
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High",
    ],
    "sparse_signed": [
        "Strong Negative",
        "Weak Negative",
        "Exactly Zero",
        "Weak Positive",
        "Strong Positive",
    ],
    "signed": [
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
    ],
}

# Regex that every condition string must satisfy: [feature_name] IS Fuzzy Value Name
_CONDITION_PATTERN = re.compile(r"^\[.+\] IS .+$")


# ---------------------------------------------------------------------------
# Strategies — shared helpers
# ---------------------------------------------------------------------------

@st.composite
def feature_name_strategy(draw: st.DrawFn) -> str:
    """
    Generate a plausible feature column name.
    Uses printable ASCII text (letters, digits, underscores) of length 1–40.
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz_0123456789"
    length = draw(st.integers(min_value=1, max_value=40))
    chars = draw(st.lists(st.sampled_from(alphabet), min_size=length, max_size=length))
    name = "".join(chars)
    if not name:
        name = "feat"
    return name


@st.composite
def valid_mode_and_gene(draw: st.DrawFn) -> tuple[str, int]:
    """
    Generate a (mode, gene) pair where gene is a valid active gene
    (i.e., 0 <= gene < num_classes, gene != dont_care).
    """
    mode = draw(st.sampled_from(_ALL_MODES))
    num_classes, dont_care = _MODE_SPECS[mode]
    gene = draw(st.integers(min_value=0, max_value=num_classes - 1))
    return mode, gene


@st.composite
def valid_chromosome_and_feature_infos(draw: st.DrawFn) -> tuple[np.ndarray, list[dict]]:
    """
    Generate a chromosome (1-D int array) and matching feature_infos list.

    Each position independently draws:
      - a mode (one of the six valid modes)
      - a gene value that is either active (0..num_classes-1) or dont_care

    The chromosome length is between 1 and 20.
    """
    n = draw(st.integers(min_value=1, max_value=20))

    feature_infos: list[dict] = []
    genes: list[int] = []

    for i in range(n):
        mode = draw(st.sampled_from(_ALL_MODES))
        num_classes, dont_care = _MODE_SPECS[mode]
        gene = draw(st.integers(min_value=0, max_value=dont_care))
        feature_name = draw(feature_name_strategy())
        if not feature_name:
            feature_name = f"feat_{i}"
        feature_infos.append({"name": feature_name, "mode": mode})
        genes.append(gene)

    chromosome = np.array(genes, dtype=np.int64)
    return chromosome, feature_infos


# ===========================================================================
# Property 8: Fuzzy Value Name Encoding Round-Trip
# Validates: Requirements 4.1, 4.2
# ===========================================================================

@given(
    feature_name=feature_name_strategy(),
    mode_and_gene=valid_mode_and_gene(),
)
@settings(
    max_examples=300,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_8a_encode_condition_starts_with_feature_name(
    feature_name: str,
    mode_and_gene: tuple[str, int],
) -> None:
    """
    **Property 8: Fuzzy Value Name Encoding Round-Trip**
    **Validates: Requirements 4.1, 4.2**

    For any valid (mode, gene) pair where gene != dont_care,
    encode_condition(feature_name, gene, mode) must return a string
    that starts with '[feature_name]'.
    """
    mode, gene = mode_and_gene
    result = encode_condition(feature_name, gene, mode)

    assert result.startswith(f"[{feature_name}]"), (
        f"encode_condition('{feature_name}', {gene}, '{mode}') = '{result}' "
        f"does not start with '[{feature_name}]'"
    )


@given(
    feature_name=feature_name_strategy(),
    mode_and_gene=valid_mode_and_gene(),
)
@settings(
    max_examples=300,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_8b_encode_condition_contains_is(
    feature_name: str,
    mode_and_gene: tuple[str, int],
) -> None:
    """
    **Property 8: Fuzzy Value Name Encoding Round-Trip**
    **Validates: Requirements 4.1, 4.2**

    For any valid (mode, gene) pair where gene != dont_care,
    encode_condition(feature_name, gene, mode) must contain ' IS '
    as the separator between the feature name and the fuzzy value name.
    """
    mode, gene = mode_and_gene
    result = encode_condition(feature_name, gene, mode)

    assert " IS " in result, (
        f"encode_condition('{feature_name}', {gene}, '{mode}') = '{result}' "
        f"does not contain ' IS '"
    )


@given(
    feature_name=feature_name_strategy(),
    mode_and_gene=valid_mode_and_gene(),
)
@settings(
    max_examples=300,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_8c_encode_condition_ends_with_valid_fuzzy_name(
    feature_name: str,
    mode_and_gene: tuple[str, int],
) -> None:
    """
    **Property 8: Fuzzy Value Name Encoding Round-Trip**
    **Validates: Requirements 4.1, 4.2**

    For any valid (mode, gene) pair where gene != dont_care,
    encode_condition(feature_name, gene, mode) must end with the
    correct fuzzy value name for that (mode, gene) pair.
    """
    mode, gene = mode_and_gene
    expected_fuzzy_name = _FUZZY_VALUE_NAMES[mode][gene]
    result = encode_condition(feature_name, gene, mode)

    assert result.endswith(expected_fuzzy_name), (
        f"encode_condition('{feature_name}', {gene}, '{mode}') = '{result}' "
        f"does not end with expected fuzzy name '{expected_fuzzy_name}'"
    )


@given(
    feature_name=feature_name_strategy(),
    mode_and_gene=valid_mode_and_gene(),
)
@settings(
    max_examples=300,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_8d_encode_condition_exact_format(
    feature_name: str,
    mode_and_gene: tuple[str, int],
) -> None:
    """
    **Property 8: Fuzzy Value Name Encoding Round-Trip**
    **Validates: Requirements 4.1, 4.2**

    For any valid (mode, gene) pair where gene != dont_care,
    encode_condition(feature_name, gene, mode) must return a string
    with the exact format '[feature_name] IS Fuzzy Value Name'.

    This checks:
      1. The full string matches the regex pattern '^\\[.+\\] IS .+$'
      2. The string equals exactly '[feature_name] IS <expected_fuzzy_name>'
    """
    mode, gene = mode_and_gene
    expected_fuzzy_name = _FUZZY_VALUE_NAMES[mode][gene]
    expected = f"[{feature_name}] IS {expected_fuzzy_name}"
    result = encode_condition(feature_name, gene, mode)

    assert _CONDITION_PATTERN.match(result), (
        f"encode_condition('{feature_name}', {gene}, '{mode}') = '{result}' "
        f"does not match pattern '^[...] IS ...$'"
    )
    assert result == expected, (
        f"encode_condition('{feature_name}', {gene}, '{mode}') = '{result}' "
        f"!= expected '{expected}'"
    )


@given(chromosome_and_infos=valid_chromosome_and_feature_infos())
@settings(
    max_examples=300,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_8e_decode_chromosome_condition_count(
    chromosome_and_infos: tuple[np.ndarray, list[dict]],
) -> None:
    """
    **Property 8: Fuzzy Value Name Encoding Round-Trip**
    **Validates: Requirements 4.1, 4.2**

    For any valid chromosome, decode_chromosome() must return a list
    whose length equals the number of non-dont_care genes in the chromosome.
    """
    chromosome, feature_infos = chromosome_and_infos

    expected_count = sum(
        1
        for gene_val, info in zip(chromosome, feature_infos)
        if int(gene_val) != _MODE_SPECS[info["mode"]][1]  # [1] = dont_care
    )

    result = decode_chromosome(chromosome, feature_infos)

    assert isinstance(result, list), (
        f"decode_chromosome() returned {type(result).__name__}, expected list"
    )
    assert len(result) == expected_count, (
        f"decode_chromosome() returned {len(result)} conditions, "
        f"expected {expected_count} (non-dont_care genes). "
        f"chromosome={chromosome.tolist()}, "
        f"modes={[info['mode'] for info in feature_infos]}"
    )


@given(chromosome_and_infos=valid_chromosome_and_feature_infos())
@settings(
    max_examples=300,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_8f_decode_chromosome_each_string_matches_pattern(
    chromosome_and_infos: tuple[np.ndarray, list[dict]],
) -> None:
    """
    **Property 8: Fuzzy Value Name Encoding Round-Trip**
    **Validates: Requirements 4.1, 4.2**

    For any valid chromosome, every string returned by decode_chromosome()
    must match the pattern '[feature_name] IS Fuzzy Value Name'.
    """
    chromosome, feature_infos = chromosome_and_infos
    result = decode_chromosome(chromosome, feature_infos)

    for condition in result:
        assert isinstance(condition, str), (
            f"decode_chromosome() returned a non-string element: {condition!r}"
        )
        assert _CONDITION_PATTERN.match(condition), (
            f"Condition '{condition}' does not match pattern '^[...] IS ...$'"
        )
        assert condition.count(" IS ") == 1, (
            f"Condition '{condition}' contains ' IS ' {condition.count(' IS ')} times, expected 1"
        )
        assert condition.startswith("["), (
            f"Condition '{condition}' does not start with '['"
        )
        bracket_close = condition.index("]")
        is_pos = condition.index(" IS ")
        assert bracket_close < is_pos, (
            f"Condition '{condition}': ']' appears after ' IS '"
        )


@given(chromosome_and_infos=valid_chromosome_and_feature_infos())
@settings(
    max_examples=300,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_8g_decode_chromosome_fuzzy_names_are_valid(
    chromosome_and_infos: tuple[np.ndarray, list[dict]],
) -> None:
    """
    **Property 8: Fuzzy Value Name Encoding Round-Trip**
    **Validates: Requirements 4.1, 4.2**

    For any valid chromosome, every condition string returned by
    decode_chromosome() must end with a fuzzy value name that is
    a valid name for the corresponding feature's mode, and the full
    condition must exactly equal '[feature_name] IS <expected_fuzzy_name>'.
    """
    chromosome, feature_infos = chromosome_and_infos
    result = decode_chromosome(chromosome, feature_infos)

    # Reconstruct expected conditions for active genes only
    expected_conditions: list[str] = []
    for gene_val, info in zip(chromosome, feature_infos):
        gene_int = int(gene_val)
        mode = info["mode"]
        _, dont_care = _MODE_SPECS[mode]
        if gene_int == dont_care:
            continue
        name = info["name"]
        fuzzy_name = _FUZZY_VALUE_NAMES[mode][gene_int]
        expected_conditions.append(f"[{name}] IS {fuzzy_name}")

    assert result == expected_conditions, (
        f"decode_chromosome() returned {result!r}, "
        f"expected {expected_conditions!r}. "
        f"chromosome={chromosome.tolist()}, "
        f"feature_infos={feature_infos}"
    )


# ===========================================================================
# Property 9: Don't-Care Sentinel Correctness
# Validates: Requirements 4.3
# ===========================================================================


@st.composite
def chromosome_with_dont_cares_strategy(draw: st.DrawFn) -> tuple[np.ndarray, list[dict]]:
    """
    Generate a chromosome array and matching feature_infos list where:
      - Each gene is either a valid class index (0..num_classes-1) or the
        dont_care sentinel (= num_classes) for its mode.
      - At least one gene is the dont_care sentinel.
      - The chromosome length is 1–20.

    Returns (chromosome, feature_infos).
    """
    n = draw(st.integers(min_value=1, max_value=20))

    genes = []
    feature_infos = []

    # Ensure at least one dont_care gene
    dont_care_positions = draw(
        st.lists(
            st.integers(min_value=0, max_value=n - 1),
            min_size=1,
            max_size=n,
            unique=True,
        )
    )
    dont_care_set = set(dont_care_positions)

    for i in range(n):
        mode = draw(st.sampled_from(_ALL_MODES))
        num_classes, dont_care = _MODE_SPECS[mode]
        feature_name = draw(feature_name_strategy())

        if i in dont_care_set:
            gene = dont_care
        else:
            # Valid gene: 0..num_classes-1
            gene = draw(st.integers(min_value=0, max_value=num_classes - 1))

        genes.append(gene)
        feature_infos.append({"name": feature_name, "mode": mode})

    chromosome = np.array(genes, dtype=np.int64)
    return chromosome, feature_infos


@st.composite
def all_active_chromosome_strategy(draw: st.DrawFn) -> tuple[np.ndarray, list[dict]]:
    """
    Generate a chromosome where NO gene is the dont_care sentinel.
    All genes are valid class indices (0..num_classes-1).
    Length is 1–20.

    Returns (chromosome, feature_infos).
    """
    n = draw(st.integers(min_value=1, max_value=20))

    genes = []
    feature_infos = []

    for _ in range(n):
        mode = draw(st.sampled_from(_ALL_MODES))
        num_classes, _ = _MODE_SPECS[mode]
        feature_name = draw(feature_name_strategy())
        gene = draw(st.integers(min_value=0, max_value=num_classes - 1))

        genes.append(gene)
        feature_infos.append({"name": feature_name, "mode": mode})

    chromosome = np.array(genes, dtype=np.int64)
    return chromosome, feature_infos


# ---------------------------------------------------------------------------
# Property 9a: dont_care sentinel equals num_classes for each mode
# ---------------------------------------------------------------------------

@given(mode=st.sampled_from(_ALL_MODES))
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_9a_dont_care_equals_num_classes(mode: str) -> None:
    """
    **Property 9: Don't-Care Sentinel Correctness — sentinel equals num_classes**
    **Validates: Requirements 4.3**

    For every valid mode, Encoder.get_dont_care(mode) must return exactly
    num_classes for that mode:
      binary → 2, ternary → 3,
      positive / sparse_positive / sparse_signed → 5,
      signed → 10
    """
    num_classes, expected_dont_care = _MODE_SPECS[mode]

    result = Encoder.get_dont_care(mode)

    assert result == expected_dont_care, (
        f"get_dont_care('{mode}') returned {result}, "
        f"expected {expected_dont_care} (= num_classes for this mode)"
    )
    assert result == num_classes, (
        f"get_dont_care('{mode}') = {result} does not equal "
        f"num_classes = {num_classes} for mode '{mode}'"
    )


# ---------------------------------------------------------------------------
# Property 9b: encode_condition raises ConfigurationError for dont_care gene
# ---------------------------------------------------------------------------

@given(
    mode=st.sampled_from(_ALL_MODES),
    feature_name=feature_name_strategy(),
)
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_9b_encode_condition_raises_for_dont_care(
    mode: str,
    feature_name: str,
) -> None:
    """
    **Property 9: Don't-Care Sentinel Correctness — encode_condition raises**
    **Validates: Requirements 4.3**

    For any mode and any feature name, calling
    encode_condition(feature_name, dont_care, mode) must raise
    ConfigurationError.  The dont_care sentinel must never produce a
    valid condition string.
    """
    dont_care = Encoder.get_dont_care(mode)

    with pytest.raises(ConfigurationError):
        Encoder.encode_condition(feature_name, dont_care, mode)


# ---------------------------------------------------------------------------
# Property 9c: dont_care is strictly greater than all valid gene values
# ---------------------------------------------------------------------------

@given(mode=st.sampled_from(_ALL_MODES))
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_9c_dont_care_strictly_greater_than_all_valid_genes(
    mode: str,
) -> None:
    """
    **Property 9: Don't-Care Sentinel Correctness — sentinel > all valid genes**
    **Validates: Requirements 4.3**

    For every valid mode, the dont_care sentinel must be strictly greater
    than every valid gene value (0..num_classes-1).  This ensures that
    dont_care cannot be confused with any encodable value.
    """
    num_classes, dont_care = _MODE_SPECS[mode]
    actual_dont_care = Encoder.get_dont_care(mode)

    # All valid genes are 0..num_classes-1
    max_valid_gene = num_classes - 1

    assert actual_dont_care > max_valid_gene, (
        f"dont_care for mode '{mode}' is {actual_dont_care}, "
        f"which is NOT strictly greater than max valid gene {max_valid_gene}"
    )

    # Also verify every individual valid gene is strictly less than dont_care
    for gene in range(num_classes):
        assert gene < actual_dont_care, (
            f"Valid gene {gene} for mode '{mode}' is not less than "
            f"dont_care sentinel {actual_dont_care}"
        )


# ---------------------------------------------------------------------------
# Property 9d: decode_chromosome skips dont_care genes
# ---------------------------------------------------------------------------

@given(data=chromosome_with_dont_cares_strategy())
@settings(
    max_examples=300,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_9d_decode_chromosome_skips_dont_care_genes(
    data: tuple[np.ndarray, list[dict]],
) -> None:
    """
    **Property 9: Don't-Care Sentinel Correctness — decode_chromosome skips dont_care**
    **Validates: Requirements 4.3**

    For any chromosome where at least one gene equals the dont_care sentinel
    for its mode, decode_chromosome must:
      1. Not raise an exception.
      2. Return fewer conditions than the total number of genes.
      3. Not include any condition for a dont_care gene position.

    The output list length must equal the number of active (non-dont_care) genes.
    """
    chromosome, feature_infos = data

    # Count active (non-dont_care) genes
    active_count = 0
    for gene_val, info in zip(chromosome, feature_infos):
        mode = info["mode"]
        dont_care = Encoder.get_dont_care(mode)
        if int(gene_val) != dont_care:
            active_count += 1

    # decode_chromosome must not raise
    result = Encoder.decode_chromosome(chromosome, feature_infos)

    # Output length must equal the number of active genes
    assert len(result) == active_count, (
        f"decode_chromosome returned {len(result)} conditions, "
        f"but expected {active_count} (active genes). "
        f"chromosome={chromosome.tolist()}, "
        f"feature_infos={feature_infos}"
    )

    # Since at least one gene is dont_care, output must be shorter than chromosome
    assert len(result) < len(chromosome), (
        f"decode_chromosome returned {len(result)} conditions for a chromosome "
        f"of length {len(chromosome)} with at least one dont_care gene. "
        f"Expected fewer conditions than genes."
    )


# ---------------------------------------------------------------------------
# Property 9e: decode_chromosome includes all active genes (no false skips)
# ---------------------------------------------------------------------------

@given(data=all_active_chromosome_strategy())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_9e_decode_chromosome_includes_all_active_genes(
    data: tuple[np.ndarray, list[dict]],
) -> None:
    """
    **Property 9: Don't-Care Sentinel Correctness — no false skips**
    **Validates: Requirements 4.3**

    For any chromosome where NO gene equals the dont_care sentinel,
    decode_chromosome must return exactly as many conditions as there are
    genes (no genes are skipped).

    This verifies that the dont_care skip logic is precise — it only skips
    genes that are exactly equal to the sentinel, not any other values.
    """
    chromosome, feature_infos = data

    result = Encoder.decode_chromosome(chromosome, feature_infos)

    assert len(result) == len(chromosome), (
        f"decode_chromosome returned {len(result)} conditions for a fully-active "
        f"chromosome of length {len(chromosome)}. Expected {len(chromosome)} conditions. "
        f"chromosome={chromosome.tolist()}, feature_infos={feature_infos}"
    )


# ---------------------------------------------------------------------------
# Property 9f: all-dont_care chromosome produces empty output
# ---------------------------------------------------------------------------

@given(mode=st.sampled_from(_ALL_MODES), n=st.integers(min_value=1, max_value=20))
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_9f_all_dont_care_chromosome_returns_empty(
    mode: str,
    n: int,
) -> None:
    """
    **Property 9: Don't-Care Sentinel Correctness — all-dont_care → empty output**
    **Validates: Requirements 4.3**

    A chromosome where every gene is the dont_care sentinel must produce
    an empty condition list from decode_chromosome.
    """
    dont_care = Encoder.get_dont_care(mode)
    chromosome = np.full(n, dont_care, dtype=np.int64)
    feature_infos = [{"name": f"feat_{i}", "mode": mode} for i in range(n)]

    result = Encoder.decode_chromosome(chromosome, feature_infos)

    assert result == [], (
        f"decode_chromosome returned {result!r} for an all-dont_care chromosome "
        f"of length {n} with mode '{mode}'. Expected []."
    )
