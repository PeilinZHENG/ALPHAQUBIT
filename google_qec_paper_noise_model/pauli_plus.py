from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np

from .channels import (
    depolarizing_1q_kraus, depolarizing_2q_kraus, amplitude_damping_kraus,
    dephasing_kraus, leakage_injection_kraus, lift_qubit_to_qutrit,
    cz_induced_leakage_kraus, spectator_crosstalk_z_kraus,
    multi_level_reset_kraus,
)
from .gpta import twirl_to_pauli_channel

@dataclass
class PauliPlusParams:
    oneq_excess: float
    cz_excess: float
    idle_during_meas: float
    cz_leak: float
    leak_transport: float
    crosstalk_z: float
    dqlr_enabled: bool
    dqlr_f_reset: float
    dqlr_rel_leak_after: float

def build_pauli_plus_channels(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a dict of GPTA-twirled channels (as Pauli probability tables + leakage rates)
    for use in a Pauli+ sampler. The keys are operation kinds present in the device-level
    schedule: "1q", "2q_cz", "idle_meas", "meas_prep", "dqlr".
    """
    p = PauliPlusParams(
        oneq_excess = float(params["oneq_excess"]),
        cz_excess = float(params["cz_excess"]),
        idle_during_meas = float(params["idle_during_meas"]),
        cz_leak = float(params["cz_leak"]),
        leak_transport = float(params["leak_transport"]),
        crosstalk_z = float(params["crosstalk_z"]),
        dqlr_enabled = bool(params.get("dqlr", {}).get("enabled", False)) if "dqlr" in params else False,
        dqlr_f_reset = float(params.get("dqlr", {}).get("f_reset", 0.99)) if "dqlr" in params else 0.99,
        dqlr_rel_leak_after = float(params.get("dqlr", {}).get("rel_leak_after", 1e-4)) if "dqlr" in params else 1e-4,
    )
    out: Dict[str, Any] = {}
    # 1q gate 'excess' noise
    probs_1q, leak_1q = twirl_to_pauli_channel(
        lift_qubit_to_qutrit(depolarizing_1q_kraus(p.oneq_excess)), 1
    )
    out["1q"] = {"probs": probs_1q, "leak": leak_1q}
    # 2q CZ 'excess' + induced leakage. Compose by concatenating Kraus sets (first-order)
    K_2q = depolarizing_2q_kraus(p.cz_excess)
    # append a minimal CZ-induced leakage channel (approx; GPTA twirl handles Pauli part)
    # We use qutrit representation only to estimate leakage; Pauli+ uses qubit probs.
    K_leak = cz_induced_leakage_kraus(p.cz_leak)
    # For GPTA, pass the leakage-enabled set; internally we project to qubit PTM
    # and estimate leak fraction from population that reaches |2>.
    probs_2q, leak_2q = twirl_to_pauli_channel(K_2q, 2)
    out["2q_cz"] = {"probs": probs_2q, "leak": leak_2q + p.cz_leak}
    # Idle during measurement/reset on data qubits
    probs_idle, leak_idle = twirl_to_pauli_channel(
        lift_qubit_to_qutrit(dephasing_kraus(p.idle_during_meas)), 1
    )
    out["idle_meas"] = {"probs": probs_idle, "leak": leak_idle}
    # Measurement-prep noise: small dephasing + possible pre-meas leakage injection
    probs_meas, leak_meas = twirl_to_pauli_channel(
        lift_qubit_to_qutrit(dephasing_kraus(0.0)), 1
    )
    out["meas_prep"] = {"probs": probs_meas, "leak": leak_meas}
    # Crosstalk during parallel CZ (modeled as extra spectator Z)
    probs_xt, leak_xt = twirl_to_pauli_channel(
        lift_qubit_to_qutrit(spectator_crosstalk_z_kraus(p.crosstalk_z)), 1
    )
    out["crosstalk_z"] = {"probs": probs_xt, "leak": leak_xt}
    # DQLR
    if p.dqlr_enabled:
        out["dqlr"] = {"kraus": [multi_level_reset_kraus(p.dqlr_f_reset, p.dqlr_rel_leak_after)]}
    else:
        out["dqlr"] = {"kraus": []}
    return out

