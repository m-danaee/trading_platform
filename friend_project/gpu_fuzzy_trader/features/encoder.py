
import numpy as np



class ConfigurationError(Exception):
    """Raised when a chromosome contains an invalid gene value (e.g. dont_care)."""



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

_DONT_CARE: dict[str, int] = {
    "binary": 2,
    "ternary": 3,
    "positive": 5,
    "sparse_positive": 5,
    "sparse_signed": 5,
    "signed": 10,
}

_VALID_MODES = frozenset(_DONT_CARE.keys())



def get_dont_care(mode: str) -> int:
    """Return the dont_care sentinel for a given mode.

    The sentinel equals num_classes for that mode:
      binary → 2, ternary → 3,
      positive / sparse_positive / sparse_signed → 5,
      signed → 10.

    Parameters
    ----------
    mode : str
        One of: "binary", "ternary", "positive", "sparse_positive",
        "sparse_signed", "signed".

    Returns
    -------
    int
        The dont_care sentinel value.

    Raises
    ------
    ConfigurationError
        If *mode* is not a recognised mode string.
    """
    if mode not in _DONT_CARE:
        raise ConfigurationError(
            f"Unknown mode '{mode}'. Valid modes: {sorted(_VALID_MODES)}"
        )
    return _DONT_CARE[mode]


def encode_condition(feature_name: str, gene: int, mode: str) -> str:
    """Return '[feature_name] IS Fuzzy Value Name' for a valid gene.

    Parameters
    ----------
    feature_name : str
        The column name of the feature (without brackets).
    gene : int
        The discretized integer value for this feature.
    mode : str
        The feature mode (determines the fuzzy value name mapping).

    Returns
    -------
    str
        Condition string of the form ``[feature_name] IS Fuzzy Value Name``.

    Raises
    ------
    ConfigurationError
        If *mode* is unknown, or if *gene* equals the dont_care sentinel
        for that mode, or if *gene* is out of range for the mode.
    """
    if mode not in _DONT_CARE:
        raise ConfigurationError(
            f"Unknown mode '{mode}'. Valid modes: {sorted(_VALID_MODES)}"
        )

    dont_care = _DONT_CARE[mode]
    if gene == dont_care:
        raise ConfigurationError(
            f"Gene value {gene} is the dont_care sentinel for mode '{mode}'. "
            "Cannot encode a dont_care gene as a condition."
        )

    names = _FUZZY_VALUE_NAMES[mode]
    if gene < 0 or gene >= len(names):
        raise ConfigurationError(
            f"Gene value {gene} is out of range for mode '{mode}' "
            f"(valid range: 0–{len(names) - 1})."
        )

    fuzzy_name = names[gene]
    return f"[{feature_name}] IS {fuzzy_name}"


def decode_chromosome(
    chromosome: np.ndarray,
    feature_infos: list[dict],
) -> list[str]:
    """Convert a chromosome array to a list of condition strings.

    Genes equal to the dont_care sentinel for their mode are silently
    skipped; only active (non-dont_care) genes produce condition strings.

    Parameters
    ----------
    chromosome : np.ndarray
        1-D integer array of length ``len(feature_infos)``.  Each element
        is the gene value for the corresponding feature.
    feature_infos : list[dict]
        Ordered list of feature descriptors.  Each dict must contain:
          - ``"name"`` (str): feature column name
          - ``"mode"`` (str): one of the six recognised mode strings

    Returns
    -------
    list[str]
        Condition strings for all active (non-dont_care) genes, in the
        same order as *feature_infos*.

    Raises
    ------
    ConfigurationError
        If any mode in *feature_infos* is unknown, or if any gene value
        is neither a valid class index nor the dont_care sentinel.
    """
    if len(chromosome) != len(feature_infos):
        raise ConfigurationError(
            f"Chromosome length {len(chromosome)} does not match "
            f"feature_infos length {len(feature_infos)}."
        )

    conditions: list[str] = []
    for gene_val, info in zip(chromosome, feature_infos):
        name = info["name"]
        mode = info["mode"]

        if mode not in _DONT_CARE:
            raise ConfigurationError(
                f"Unknown mode '{mode}' for feature '{name}'. "
                f"Valid modes: {sorted(_VALID_MODES)}"
            )

        dont_care = _DONT_CARE[mode]
        gene_int = int(gene_val)

        if gene_int == dont_care:
            continue

        names = _FUZZY_VALUE_NAMES[mode]
        if gene_int < 0 or gene_int >= len(names):
            raise ConfigurationError(
                f"Gene value {gene_int} is out of range for mode '{mode}' "
                f"on feature '{name}' (valid range: 0–{len(names) - 1})."
            )

        conditions.append(f"[{name}] IS {names[gene_int]}")

    return conditions



class Encoder:
    """Stateless encoder that maps gene values to fuzzy condition strings."""

    @staticmethod
    def get_dont_care(mode: str) -> int:
        """See module-level :func:`get_dont_care`."""
        return get_dont_care(mode)

    @staticmethod
    def encode_condition(feature_name: str, gene: int, mode: str) -> str:
        """See module-level :func:`encode_condition`."""
        return encode_condition(feature_name, gene, mode)

    @staticmethod
    def decode_chromosome(
        chromosome: np.ndarray,
        feature_infos: list[dict],
    ) -> list[str]:
        """See module-level :func:`decode_chromosome`."""
        return decode_chromosome(chromosome, feature_infos)
