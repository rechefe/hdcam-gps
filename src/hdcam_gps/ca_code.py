"""GPS L1 C/A code generation.

The C/A code is the product of two 10-stage maximal-length LFSRs (G1 and G2).
Each satellite uses the same G1 sequence and a PRN-specific delay of G2, which
is realized here by tapping two of the G2 stages.
"""

import numpy as np

from hdcam_gps.acq_base import CHIPS_PER_CODE, CODE_PERIOD_S

# PRN -> the pair of G2 stages (1-indexed) whose modulo-2 sum forms the
# satellite specific G2 delay. IS-GPS-200, table 3-Ia.
G2_TAPS: dict[int, tuple[int, int]] = {
    1: (2, 6),
    2: (3, 7),
    3: (4, 8),
    4: (5, 9),
    5: (1, 9),
    6: (2, 10),
    7: (1, 8),
    8: (2, 9),
    9: (3, 10),
    10: (2, 3),
    11: (3, 4),
    12: (5, 6),
    13: (6, 7),
    14: (7, 8),
    15: (8, 9),
    16: (9, 10),
    17: (1, 4),
    18: (2, 5),
    19: (3, 6),
    20: (4, 7),
    21: (5, 8),
    22: (6, 9),
    23: (1, 3),
    24: (4, 6),
    25: (5, 7),
    26: (6, 8),
    27: (7, 9),
    28: (8, 10),
    29: (1, 6),
    30: (2, 7),
    31: (3, 8),
    32: (4, 9),
}


def ca_code(prn_idx: int) -> np.ndarray:
    """Generates the 1023 chip C/A code of a single satellite.

    Args:
        prn_idx (int): The PRN number of the satellite, 1 to 32.

    Returns:
        np.ndarray: The code as +1/-1 chips, of length CHIPS_PER_CODE.
    """
    assert (
        prn_idx in G2_TAPS
    ), f"PRN index must be one of {sorted(G2_TAPS)}, but got {prn_idx}."
    tap_a, tap_b = G2_TAPS[prn_idx]

    g1 = np.ones(10, dtype=np.int8)
    g2 = np.ones(10, dtype=np.int8)
    chips = np.empty(CHIPS_PER_CODE, dtype=np.int8)

    for chip in range(CHIPS_PER_CODE):
        chips[chip] = g1[9] ^ g2[tap_a - 1] ^ g2[tap_b - 1]
        g1_feedback = g1[2] ^ g1[9]
        g2_feedback = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1[1:] = g1[:-1]
        g1[0] = g1_feedback
        g2[1:] = g2[:-1]
        g2[0] = g2_feedback

    return 1 - 2 * chips  # 0 -> +1, 1 -> -1


def sampled_ca_code(prn_idx: int, fs_hz: float) -> np.ndarray:
    """Samples one C/A code period of a satellite at the given sampling frequency.

    The code is held at each chip value (nearest chip, no interpolation). Longer
    replicas are just this period tiled, so the caller can np.tile the result.

    Args:
        prn_idx (int): The PRN number of the satellite.
        fs_hz (float): The sampling frequency in Hz.

    Returns:
        np.ndarray: The sampled code as +1/-1 values, of length
            int(fs_hz * CODE_PERIOD_S).
    """
    n_samples = int(fs_hz * CODE_PERIOD_S)
    chip_rate_hz = CHIPS_PER_CODE / CODE_PERIOD_S
    chip_index = (np.arange(n_samples) * chip_rate_hz / fs_hz).astype(int)
    return ca_code(prn_idx)[chip_index]
