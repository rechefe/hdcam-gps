"""Directed tests for the C/A code generator.

Every PRN is checked against two references that are independent of the
implementation under test:

* FIRST_10_CHIPS_OCTAL - the "first 10 chips" column of IS-GPS-200 table 3-I.
* reference_ca_code    - the alternative textbook construction, G1 xor a
                         delayed G2, using the code delay column of the same
                         table instead of the G2 tap pair. LAST_10_CHIPS_OCTAL
                         is the tail of that construction.
"""

import numpy as np
import pytest

from hdcam_gps.acq_base import CHIPS_PER_CODE, CODE_PERIOD_S
from hdcam_gps.ca_code import G2_TAPS, ca_code, sampled_ca_code

ALL_PRN_IDX = sorted(G2_TAPS)

# IS-GPS-200 table 3-I, first 10 chips, as the octal value of those 10 bits.
FIRST_10_CHIPS_OCTAL = {
    1: 0o1440,
    2: 0o1620,
    3: 0o1710,
    4: 0o1744,
    5: 0o1133,
    6: 0o1455,
    7: 0o1131,
    8: 0o1454,
    9: 0o1626,
    10: 0o1504,
    11: 0o1642,
    12: 0o1750,
    13: 0o1764,
    14: 0o1772,
    15: 0o1775,
    16: 0o1776,
    17: 0o1156,
    18: 0o1467,
    19: 0o1633,
    20: 0o1715,
    21: 0o1746,
    22: 0o1763,
    23: 0o1063,
    24: 0o1706,
    25: 0o1743,
    26: 0o1761,
    27: 0o1770,
    28: 0o1774,
    29: 0o1127,
    30: 0o1453,
    31: 0o1625,
    32: 0o1712,
}

# Tail of the G1 xor delayed G2 construction, same encoding as above.
LAST_10_CHIPS_OCTAL = {
    1: 0o0420,
    2: 0o0310,
    3: 0o1044,
    4: 0o1522,
    5: 0o1162,
    6: 0o1571,
    7: 0o1144,
    8: 0o0562,
    9: 0o1371,
    10: 0o1000,
    11: 0o0500,
    12: 0o1460,
    13: 0o1730,
    14: 0o1654,
    15: 0o1626,
    16: 0o0613,
    17: 0o1700,
    18: 0o0640,
    19: 0o0220,
    20: 0o1010,
    21: 0o1504,
    22: 0o1742,
    23: 0o0400,
    24: 0o1120,
    25: 0o1550,
    26: 0o1764,
    27: 0o1672,
    28: 0o0635,
    29: 0o1020,
    30: 0o0510,
    31: 0o0344,
    32: 0o1062,
}

# IS-GPS-200 table 3-I, G2 code delay in chips.
G2_DELAY_CHIPS = {
    1: 5,
    2: 6,
    3: 7,
    4: 8,
    5: 17,
    6: 18,
    7: 139,
    8: 140,
    9: 141,
    10: 251,
    11: 252,
    12: 254,
    13: 255,
    14: 256,
    15: 257,
    16: 258,
    17: 469,
    18: 470,
    19: 471,
    20: 472,
    21: 473,
    22: 474,
    23: 509,
    24: 512,
    25: 513,
    26: 514,
    27: 515,
    28: 516,
    29: 859,
    30: 860,
    31: 861,
    32: 862,
}


def as_bits(code: np.ndarray) -> np.ndarray:
    """Maps the +1/-1 chips back to the 0/1 convention of the standard."""
    return (1 - code.astype(int)) // 2


def as_octal(bits: np.ndarray) -> int:
    """Reads a run of bits as one octal/binary number, MSB first."""
    return int("".join(str(bit) for bit in bits), 2)


def m_sequence(taps: list[int]) -> np.ndarray:
    """Generates a 1023 chip maximal length sequence from an all-ones register."""
    register = np.ones(10, dtype=int)
    sequence = np.empty(CHIPS_PER_CODE, dtype=int)
    for chip in range(CHIPS_PER_CODE):
        sequence[chip] = register[9]
        feedback = 0
        for tap in taps:
            feedback ^= register[tap - 1]
        register[1:] = register[:-1]
        register[0] = feedback
    return sequence


def reference_ca_code(prn_idx: int) -> np.ndarray:
    """Builds the code as G1 xor a delayed G2, in the 0/1 convention."""
    g1 = m_sequence([3, 10])
    g2 = m_sequence([2, 3, 6, 8, 9, 10])
    return g1 ^ np.roll(g2, G2_DELAY_CHIPS[prn_idx])


# --------------------------------------------------------------------------
# ca_code - per PRN expectations
# --------------------------------------------------------------------------


def test_all_32_prn_indices_are_defined():
    assert ALL_PRN_IDX == list(range(1, 33))


@pytest.mark.parametrize("prn_idx", ALL_PRN_IDX)
def test_code_has_the_right_length_and_alphabet(prn_idx):
    code = ca_code(prn_idx)
    assert len(code) == CHIPS_PER_CODE
    assert set(np.unique(code)) == {-1, 1}


@pytest.mark.parametrize("prn_idx", ALL_PRN_IDX)
def test_first_10_chips_match_the_standard(prn_idx):
    bits = as_bits(ca_code(prn_idx))
    assert as_octal(bits[:10]) == FIRST_10_CHIPS_OCTAL[prn_idx]


@pytest.mark.parametrize("prn_idx", ALL_PRN_IDX)
def test_last_10_chips_match_the_reference(prn_idx):
    bits = as_bits(ca_code(prn_idx))
    assert as_octal(bits[-10:]) == LAST_10_CHIPS_OCTAL[prn_idx]


@pytest.mark.parametrize("prn_idx", ALL_PRN_IDX)
def test_whole_code_matches_the_delayed_g2_construction(prn_idx):
    assert np.array_equal(as_bits(ca_code(prn_idx)), reference_ca_code(prn_idx))


@pytest.mark.parametrize("prn_idx", ALL_PRN_IDX)
def test_code_is_balanced(prn_idx):
    # 1023 chips: 512 of one sign, 511 of the other, so the sum is -1.
    assert ca_code(prn_idx).astype(int).sum() == -1


def test_codes_of_different_prn_indices_differ():
    codes = {prn_idx: ca_code(prn_idx).tobytes() for prn_idx in ALL_PRN_IDX}
    assert len(set(codes.values())) == len(ALL_PRN_IDX)


def test_generation_is_deterministic():
    assert np.array_equal(ca_code(19), ca_code(19))


@pytest.mark.parametrize("bad_prn_idx", [0, -1, 33, 100])
def test_ca_code_rejects_an_unknown_prn_index(bad_prn_idx):
    with pytest.raises(AssertionError):
        ca_code(bad_prn_idx)


# --------------------------------------------------------------------------
# ca_code - Gold code correlation properties
# --------------------------------------------------------------------------


@pytest.mark.parametrize("prn_idx", [1, 7, 19, 32])
def test_autocorrelation_takes_only_the_three_gold_values(prn_idx):
    code = ca_code(prn_idx).astype(int)
    shifted = np.array([np.dot(code, np.roll(code, shift)) for shift in range(1, 1023)])
    assert np.dot(code, code) == CHIPS_PER_CODE  # the peak
    assert set(np.unique(shifted)) <= {-65, -1, 63}


@pytest.mark.parametrize(("prn_a", "prn_b"), [(1, 2), (1, 19), (7, 32), (24, 25)])
def test_cross_correlation_stays_at_the_gold_values(prn_a, prn_b):
    code_a = ca_code(prn_a).astype(int)
    code_b = ca_code(prn_b).astype(int)
    products = np.array(
        [np.dot(code_a, np.roll(code_b, shift)) for shift in range(1023)]
    )
    assert set(np.unique(products)) <= {-65, -1, 63}


# --------------------------------------------------------------------------
# sampled_ca_code
# --------------------------------------------------------------------------


def test_sampled_code_has_one_code_period_of_samples():
    assert len(sampled_ca_code(1, 4e6)) == 4000
    assert len(sampled_ca_code(1, 2.046e6)) == 2046
    assert len(sampled_ca_code(1, 1.023e6)) == 1023


def test_sampling_at_the_chip_rate_reproduces_the_chips():
    for prn_idx in (1, 5, 32):
        assert np.array_equal(sampled_ca_code(prn_idx, 1.023e6), ca_code(prn_idx))


def test_sampling_at_twice_the_chip_rate_repeats_every_chip():
    sampled = sampled_ca_code(11, 2.046e6)
    assert np.array_equal(sampled, np.repeat(ca_code(11), 2))


def test_sampled_code_keeps_the_plus_minus_one_alphabet():
    sampled = sampled_ca_code(3, 4e6)
    assert set(np.unique(sampled)) == {-1, 1}


def test_sampled_code_starts_and_ends_on_the_right_chip():
    code = ca_code(9)
    sampled = sampled_ca_code(9, 4e6)
    assert sampled[0] == code[0]
    assert sampled[-1] == code[-1]


def test_sample_count_follows_the_code_period():
    fs_hz = 4e6
    assert len(sampled_ca_code(1, fs_hz)) == int(fs_hz * CODE_PERIOD_S)


def test_sampled_code_rejects_an_unknown_prn_index():
    with pytest.raises(AssertionError):
        sampled_ca_code(0, 4e6)
