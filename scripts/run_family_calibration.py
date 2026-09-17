"""Runs the phase 3 calibration of docs/CAM_FAMILY_STUDY.md over the calibration skies.

Section 4.5: every surviving family and the FFT reference go through one
protocol on set A, and the setting each is compared at afterwards is frozen into
docs/calibration.json. Phase 4 reads that file and never sees set A again.

    uv run python scripts/run_family_calibration.py

The run takes about three hours and needs the gps-sdr-sim submodule, so the
result is checked in. Families are done cheapest first and the JSON is rewritten
after each one, so a run that is cut short still leaves everything it finished.

Two families are absent and both absences are deliberate. The differential
family was killed at phase 1 by rule (i) - its d' of 1.15 is below half the
baseline's - so calibrating it would be measuring a design the study has already
declined; it is written into the JSON as null with the reason, because "dead" is
a result and a missing key is not. The code-only exact mixer is an upper bound
rather than a design, so it is not a family to freeze a setting for either.

The segmented family goes through RefinedCamCalibrator rather than
CamCalibrator: it names a PRN and a code phase and nothing else, and section 4.1
counts a detection at the wrong Doppler as a miss and a false alarm at once, so
without the second stage in the loop every setting would be rejected for a
reason that has nothing to do with the setting.
"""

import json
import time
from pathlib import Path

from hdcam_gps.acq_base import AcqConfig
from hdcam_gps.calibrate import (
    DEFAULT_TARGET_PFA,
    CamCalibrator,
    FrozenSetting,
    OperatingPoint,
    PeakRatioCalibrator,
    RefinedCamCalibrator,
    bank_records,
    match_false_alarm,
    run_calibration,
    save_calibration,
)
from hdcam_gps.code_only_acq import CodeOnlyHdCamClassifier
from hdcam_gps.evaluate import EvalConfig
from hdcam_gps.fft_acq import FftAcqClassifier
from hdcam_gps.hdcam_acq import OneBitHdCamClassifier
from hdcam_gps.scenarios import ScenarioBank
from hdcam_gps.segmented_acq import SegmentedHdCamClassifier
from hdcam_gps.thermometer_acq import ThermometerHdCamClassifier

RESULTS: Path = Path(__file__).resolve().parents[1] / "docs" / "calibration.json"

# The headline configuration of section 6, as phase 1 screened it.
CONFIG = AcqConfig(
    fs_hz=1.023e6,
    prn_list=tuple(range(1, 33)),
    doppler_min_hz=-5000.0,
    doppler_max_hz=5000.0,
    doppler_step_hz=500.0,
    n_codes=10,
)
N_CALIBRATION_SKIES: int = 20  # Set A of section 4.5, the same 20 phase 1 screened
SCALINGS: tuple[float, ...] = (54.0, 51.0, 48.0, 45.0, 42.0, 39.0, 36.0)
TARGET = OperatingPoint(
    cn0_dbhz=40.0,
    max_pmd=0.10,
    max_pfa=DEFAULT_TARGET_PFA,  # The honest target; 1e-4 is not demonstrated
    code_phase_tolerance=1,
    doppler_bins=1.0,
    band_dbhz=(38.0, 42.0),
)
# Section 4.5 asks for the pick to be gated on the 95 percent bound. With zero
# events in n records that bound is 3/n for every setting, so on 140 calibration
# records it is 2.1e-2 and no setting of any family could clear 1e-2 - the gate
# would be measuring the record count rather than the design. The pick is made
# on the measured rate instead, the bound is printed beside it, and the bound is
# what the evaluation set's 280 records are for.
PICK_ON: str = "measured"
DEAD: dict[str, str] = {
    "differential": (
        "killed at phase 1 by rule (i): d' of 1.15 in the 45 dB-Hz bin is below "
        "half the baseline's 6.25"
    )
}


def calibrators(config: AcqConfig) -> dict:
    """Every survivor and the reference, cheapest first.

    Cheapest first so that a run which is cut short has lost the least. The
    order is the one the per record costs measured on this machine imply: the
    reference is half a second, the baseline nine, the thermometer twelve, the
    segmented seventy and code-only a minute.

    Args:
        config (AcqConfig): The configuration to build them for.

    Returns:
        dict: Name to a replayer run_calibration can drive.
    """
    return {
        "fft reference": PeakRatioCalibrator(FftAcqClassifier(config)),
        "baseline": CamCalibrator(OneBitHdCamClassifier(config, search_mode="table")),
        "thermometer": CamCalibrator(
            ThermometerHdCamClassifier(config, search_mode="table")
        ),
        "segmented": RefinedCamCalibrator(
            SegmentedHdCamClassifier(config, search_mode="table")
        ),
        "code-only (quadrant)": CamCalibrator(
            CodeOnlyHdCamClassifier(config, mixer="quadrant", search_mode="table")
        ),
    }


def meta(n_records: int, seconds: float, refused: list) -> dict:
    """What the run was, written beside the settings it produced.

    A null setting in the JSON says a family has none; this says why, which is
    two different reasons - never calibrated, or calibrated and no threshold
    held the rate.

    Args:
        n_records (int): Records every family was calibrated on.
        seconds (float): Wall clock so far.
        refused (list): Families where no setting met the target.

    Returns:
        dict: The configuration, the split and the target.
    """
    return {
        "fs_hz": CONFIG.fs_hz,
        "n_prn": len(CONFIG.prn_list),
        "doppler_min_hz": CONFIG.doppler_min_hz,
        "doppler_max_hz": CONFIG.doppler_max_hz,
        "doppler_step_hz": CONFIG.doppler_step_hz,
        "n_codes": CONFIG.n_codes,
        "n_calibration_skies": N_CALIBRATION_SKIES,
        "scalings_dbhz": list(SCALINGS),
        "n_records": n_records,
        "target_pfa": TARGET.max_pfa,
        "picked_on": PICK_ON,
        "band_dbhz": list(TARGET.band_dbhz),
        "code_phase_tolerance": TARGET.code_phase_tolerance,
        "doppler_bins": TARGET.doppler_bins,
        "seconds": seconds,
        "set": "A (calibration skies); nothing here is measured on set B",
        "not_calibrated": DEAD,
        "no_setting_met_the_target": list(refused),
    }


def main():
    """Calibrates every survivor and the reference on set A, and freezes the pick."""
    sweep = EvalConfig(
        n_scenarios=N_CALIBRATION_SKIES,
        cn0_dbhz=SCALINGS,
        backend="simulator",
        progress=True,
    )
    print(f"building {N_CALIBRATION_SKIES} skies at {len(SCALINGS)} scalings")
    bank = ScenarioBank.build(CONFIG, sweep)
    records = bank_records(bank)
    print(
        f"{len(records)} records, {bank.satellite_cn0_dbhz().size} satellites, "
        f"target Pfa {TARGET.max_pfa:g} per acquisition, picked on the "
        f"{PICK_ON} rate"
    )

    settings: dict = {name: None for name in DEAD}
    refused: list = []
    started = time.perf_counter()
    for name, replayer in calibrators(CONFIG).items():
        family_started = time.perf_counter()
        result = run_calibration(replayer, records, TARGET, progress=True)
        picked = match_false_alarm(result, TARGET.max_pfa, bound=PICK_ON)
        seconds = time.perf_counter() - family_started
        print(f"\n{name}  ({seconds:.0f} s)")
        print(result.table())
        print(
            f"  {result.trials_needed} records would be needed for a zero event "
            f"95% bound to reach {TARGET.max_pfa:g}; this ran on {len(records)}"
        )
        if picked is None:
            print(
                f"  no setting held Pfa below {TARGET.max_pfa:g}: "
                "dead by kill rule (ii)"
            )
            settings[name] = None
            refused.append(name)
        else:
            frozen = FrozenSetting.from_candidate(name, picked, TARGET.max_pfa)
            print(
                f"  picked {frozen.setting}: Pfa {frozen.pfa:.4f} "
                f"(95% {frozen.pfa_upper:.4f}), Pd {frozen.pd:.3f}, "
                f"Pd in {TARGET.band_dbhz[0]:.0f}-{TARGET.band_dbhz[1]:.0f} dB-Hz "
                f"{frozen.pd_in_band:.3f} over {frozen.n_band_satellites} satellites"
            )
            settings[name] = frozen
        save_calibration(
            RESULTS,
            settings,
            meta(len(records), time.perf_counter() - started, refused),
        )
        print(f"  wrote {RESULTS}", flush=True)

    for name, reason in DEAD.items():
        print(f"{name}: not calibrated - {reason}")
    print(
        json.dumps(
            meta(len(records), time.perf_counter() - started, refused), indent=2
        )
    )


if __name__ == "__main__":
    main()
