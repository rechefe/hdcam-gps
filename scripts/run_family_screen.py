"""Runs the phase 1 screen of docs/CAM_FAMILY_STUDY.md over the calibration skies.

Writes docs/screen_results.json, which notebooks/family_screen.ipynb plots. The
run takes about three quarters of an hour and needs the gps-sdr-sim submodule,
so the result is checked in and the notebook reads it rather than repeating it.

    uv run python scripts/run_family_screen.py

The code-only family is the long pole at 25 seconds a record, because moving the
CFO into the query multiplies the searches by the 21 Doppler bins. Its exact
mixer is an upper bound rather than a design, so it is screened on fewer records.
"""

import json
import time
from pathlib import Path

from hdcam_gps.acq_base import AcqConfig
from hdcam_gps.calibrate import bank_records
from hdcam_gps.code_only_acq import CodeOnlyHdCamClassifier
from hdcam_gps.diff_acq import DifferentialHdCamClassifier
from hdcam_gps.evaluate import EvalConfig
from hdcam_gps.hdcam_acq import OneBitHdCamClassifier
from hdcam_gps.scenarios import ScenarioBank
from hdcam_gps.screen import screen
from hdcam_gps.segmented_acq import SegmentedHdCamClassifier
from hdcam_gps.thermometer_acq import ThermometerHdCamClassifier

RESULTS: Path = Path(__file__).resolve().parents[1] / "docs" / "screen_results.json"

# The headline configuration of section 6: 32 PRNs, +-5 kHz at 500 Hz, 10 ms.
CONFIG = AcqConfig(
    fs_hz=1.023e6,
    prn_list=tuple(range(1, 33)),
    doppler_min_hz=-5000.0,
    doppler_max_hz=5000.0,
    doppler_step_hz=500.0,
    n_codes=10,
)
N_CALIBRATION_SKIES: int = 20  # Set A of section 4.5
SCALINGS: tuple[float, ...] = (54.0, 45.0, 36.0)  # Record averages, the sampling design
UPPER_BOUND_RECORDS: int = 12  # For families reported as a bound rather than a design


def families(config: AcqConfig) -> dict:
    """The five families and the code-only upper bound, all on the table path.

    Args:
        config (AcqConfig): The configuration to build them for.

    Returns:
        dict: Name to classifier, in the order the tables print.
    """
    return {
        "baseline": OneBitHdCamClassifier(config, search_mode="table"),
        "code-only (quadrant)": CodeOnlyHdCamClassifier(
            config, mixer="quadrant", search_mode="table"
        ),
        "code-only (exact)": CodeOnlyHdCamClassifier(
            config, mixer="exact", search_mode="table"
        ),
        "thermometer": ThermometerHdCamClassifier(config, search_mode="table"),
        "segmented": SegmentedHdCamClassifier(config, search_mode="table"),
        "differential": DifferentialHdCamClassifier(config, search_mode="table"),
    }


def as_json(separation, seconds: float) -> dict:
    """Flattens a Separation into what the notebook needs to plot it.

    The per look distances are not kept: they are half a megabyte a family, and
    every number the study reads off them is already in the bins.

    Args:
        separation (Separation): One family's screen.
        seconds (float): Wall clock time the screen took.

    Returns:
        dict: The floors, the bins and the geometry.
    """
    return {
        "family": separation.family,
        "n_rows": separation.n_rows,
        "n_columns": separation.n_columns,
        "n_records": separation.n_records,
        "n_looks": int(separation.true_distance.size),
        "wrong_mean": separation.wrong_mean,
        "wrong_sd": separation.wrong_sd,
        "n_wrong": separation.n_wrong,
        "noise_mean": separation.noise_mean,
        "noise_sd": separation.noise_sd,
        "seconds": seconds,
        "bins": [
            {
                "cn0_dbhz": entry.cn0_dbhz,
                "n_looks": entry.n_looks,
                "true_mean": entry.true_mean,
                "true_percentile": entry.true_percentile,
                "d_prime": entry.d_prime,
                "tolerance_fraction": entry.tolerance_fraction,
                "resolution_fraction": entry.resolution_fraction,
                "resolution_sigma": entry.resolution_sigma,
            }
            for entry in separation.bins()
        ],
    }


def main():
    """Screens every family on set A and writes the result."""
    sweep = EvalConfig(
        n_scenarios=N_CALIBRATION_SKIES,
        cn0_dbhz=SCALINGS,
        backend="simulator",
        progress=True,
    )
    print(f"building {N_CALIBRATION_SKIES} skies at {len(SCALINGS)} scalings")
    bank = ScenarioBank.build(CONFIG, sweep)
    records = bank_records(bank)
    print(f"{len(records)} records, {bank.satellite_cn0_dbhz().size} satellites")

    results = {
        "config": {
            "fs_hz": CONFIG.fs_hz,
            "n_prn": len(CONFIG.prn_list),
            "doppler_min_hz": CONFIG.doppler_min_hz,
            "doppler_max_hz": CONFIG.doppler_max_hz,
            "doppler_step_hz": CONFIG.doppler_step_hz,
            "n_codes": CONFIG.n_codes,
            "n_calibration_skies": N_CALIBRATION_SKIES,
            "scalings_dbhz": list(SCALINGS),
        },
        "families": [],
    }
    for name, classifier in families(CONFIG).items():
        subset = records if "exact" not in name else records[:UPPER_BOUND_RECORDS]
        started = time.perf_counter()
        separation = screen(classifier, subset, family=name, progress=True)
        seconds = time.perf_counter() - started
        results["families"].append(as_json(separation, seconds))
        print(separation.table(), flush=True)
        print(f"  {seconds:.0f} s\n", flush=True)

    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
