
from typing import Any, Dict, List, Tuple


import math

import stim
from google_qec_paper_noise_model.gpt import gpt_single_qubit, amp_phase_kraus


class PauliPlusSimulator:
    def __init__(self, config: Dict, basis: str):
        self.config = config
        self.basis = basis.upper()
        if self.basis not in ("X", "Z"):
            raise ValueError("basis must be 'X' or 'Z'")

    def __init__(self, config: Dict, basis: str):

        self.distance = config.get("distance")
        self.rounds = config.get("rounds")
        if self.distance is None or self.rounds is None:
            raise ValueError("config must include 'distance' and 'rounds'")

        self.depolarization = config.get("depolarization", 0.001)
        self.leakage_rate = config.get("leakage_rate", 0.01)
        self.cross_talk = config.get("cross_talk", 0.002)


        self.circuit = self._build_base_circuit(config, self.basis)
        # Backward compatible noise attachment after two-qubit gates
        self.circuit = self._attach_noise_to_two_qubit_gates(self.circuit)
        # paper-aligned injection is applied explicitly via apply_paper_aligned_noise(...)

        self.basis = basis.upper()
        if self.basis not in ("X", "Z"):
            raise ValueError("basis must be 'X' or 'Z'")

        self.circuit = self._build_noisy_circuit()
        # If configs/paper_aligned.yaml was used or keys exist, allow in-place instrumentation.
        if any(
            k in config
            for k in (
                "T1_us",
                "Tphi_us",
                "p_cz_crosstalk_ZZ",
                "p_cz_swap_like",
                "p_cz_excess",
            )
        ):
            # Do nothing here; paper-aligned code calls apply_paper_aligned_noise explicitly.
            pass


    def _build_base_circuit(self, config: Dict, basis: str) -> stim.Circuit:
        circuit = stim.Circuit.generated(
            f"surface_code:rotated_memory_{basis.lower()}",
            rounds=self.rounds,
            distance=self.distance,
        )
        return circuit

    def _attach_noise_to_two_qubit_gates(self, circuit: stim.Circuit) -> stim.Circuit:
        """Insert leakage and cross-talk noise after each two-qubit gate."""
        noisy = stim.Circuit()
        for inst in circuit:
            noisy.append(inst)
            if inst.name in ("CX", "CZ"):
                targets = [t.value for t in inst.targets_copy()]
                noisy.append_operation(
                    "PAULI_CHANNEL_2",
                    targets,
                    [
                        self.leakage_rate / 3,
                        self.leakage_rate / 3,
                        self.leakage_rate / 3,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                )
                noisy.append_operation("DEPOLARIZE2", targets, self.cross_talk)
        return noisy


 
    def apply_paper_aligned_noise(self, config: Dict) -> None:
        """
        Instrument the existing Stim circuit with paper-aligned GPT and correlated errors.
        - 1Q channels: GPT( T1, Tphi ) -> PAULI_CHANNEL_1(pX,pY,pZ) after 1Q gates.
        - 2Q CZ windows: inject correlated ZZ & swap-like via CORRELATED_ERROR at TICK boundaries.
        - Residuals: add 1Q PAULI_CHANNEL_1 on each CZ participant (p_1q_excess folded here).
        Detector layout (DETECTOR/OBSERVABLE_INCLUDE/coords) is untouched.

        """
        cfg = config
        cycle_ns = float(cfg.get("cycle_ns", 1100.0))
        dt_us = cycle_ns / 1000.0
        T1_us = float(cfg.get("T1_us", 68.0))
        Tphi_us = float(cfg.get("Tphi_us", 89.0))
        p_1q_excess = float(cfg.get("p_1q_excess", 5e-4))
        p_idle_excess = float(cfg.get("p_idle_excess", 5e-4))
        twirl_idles_each_tick = bool(cfg.get("twirl_idles_each_tick", False))
        twirl_after_1q_gates = bool(cfg.get("twirl_after_1q_gates", True))
 
        p_readout = float(cfg.get("p_readout", 3e-3))
        p_reset = float(cfg.get("p_reset", 3e-3))

        # GPT-twirled 1q probabilities from amplitude+phase damping over one cycle
        p1 = gpt_single_qubit(amp_phase_kraus(dt_us=dt_us, T1_us=T1_us, Tphi_us=Tphi_us))
        px, py, pz = p1.get("X", 0.0), p1.get("Y", 0.0), p1.get("Z", 0.0)
        # Fold 'excess' into evenly split XYZ
        px += p_1q_excess / 3.0
        py += p_1q_excess / 3.0
        pz += p_1q_excess / 3.0

        # CZ correlated terms (parallel window)
        p_cz_zz = float(cfg.get("p_cz_crosstalk_ZZ", 5e-4))
        p_cz_swap = float(cfg.get("p_cz_swap_like", 5e-4))
        p_cz_excess = float(cfg.get("p_cz_excess", 1e-3))

        ONE_Q_GATES = {
            "H", "X", "Y", "Z", "S", "SQRT_X", "SQRT_Y", "RX", "RY", "RZ",
            # common aliases:
            "H_XZ", "H_YZ", "T", "SQRT_Z"
        }
        TWO_Q_AS_CZ = {"CZ", "CNOT", "CX"}

        new_circuit = stim.Circuit()
        current_cz_pairs: List[Tuple[int, int]] = []
        seen_qubits: set[int] = set()

        def add_pauli_ch1(q: int, px_: float, py_: float, pz_: float):
            if px_ <= 0 and py_ <= 0 and pz_ <= 0:
                return
            new_circuit.append_operation("PAULI_CHANNEL_1", [q], [px_, py_, pz_])

        def inject_cz_pair(a: int, b: int):
 

        # GPT-twirled 1q probabilities from amplitude+phase damping over one cycle
        p1 = gpt_single_qubit(amp_phase_kraus(dt_us=dt_us, T1_us=T1_us, Tphi_us=Tphi_us))
        pX, pY, pZ = p1.get("X", 0.0), p1.get("Y", 0.0), p1.get("Z", 0.0)
        # Fold 'excess' into evenly split XYZ
        pX += p_1q_excess / 3.0
        pY += p_1q_excess / 3.0
        pZ += p_1q_excess / 3.0

        # CZ correlated terms
        p_cz_zz = float(cfg.get("p_cz_crosstalk_ZZ", 5e-4))
        p_cz_swap = float(cfg.get("p_cz_swap_like", 5e-4))
        p_cz_excess = float(cfg.get("p_cz_excess", 1e-3))  # additional 1q around CZ

        ONE_Q_GATES = {
            "H",
            "X",
            "Y",
            "Z",
            "S",
            "SQRT_X",
            "SQRT_Y",
            "RX",
            "RY",
            "RZ",
            # common aliases present in Stim circuits:
            "H_XZ",
            "H_YZ",
            "T",
            "SQRT_Z",
        }
        TWO_Q_CZ = {"CZ", "CNOT", "CX"}  # treat CNOT like CZ for error-injection purposes

        new_circuit = stim.Circuit()
        current_cz_pairs: List[Tuple[int, int]] = []
        all_qubits_seen: set[int] = set()

        def append_pauli_channel_1(q: int, px: float, py: float, pz: float):
            # Stim's PAULI_CHANNEL_1 models a Pauli-mixture; keeps semantics exact vs X_ERROR/Y_ERROR/Z_ERROR
            if px <= 0 and py <= 0 and pz <= 0:
                return
            new_circuit.append_operation("PAULI_CHANNEL_1", [q], [px, py, pz])

        def inject_cz_correlated(a: int, b: int):
 
            # ZZ crosstalk
            if p_cz_zz > 0.0:
                new_circuit.append_operation(
                    "CORRELATED_ERROR",
                    [stim.target_pauli_z(a), stim.target_pauli_z(b)],
                    [p_cz_zz],
                )
 
            # swap-like: approximate with equal XX and YY
 
            if p_cz_swap > 0.0:
                new_circuit.append_operation(
                    "CORRELATED_ERROR",
                    [stim.target_pauli_x(a), stim.target_pauli_x(b)],
                    [0.5 * p_cz_swap],
                )
                new_circuit.append_operation(
                    "CORRELATED_ERROR",
                    [stim.target_pauli_y(a), stim.target_pauli_y(b)],
                    [0.5 * p_cz_swap],
                )
 
            # residual 1q around CZ
            if p_cz_excess > 0.0:
                s = p_cz_excess / 3.0
                add_pauli_ch1(a, s, s, s)
                add_pauli_ch1(b, s, s, s)

        # Iterate original circuit and instrument
        for inst in self.circuit:
            name = inst.name
            targs = inst.targets_copy()
            gargs = inst.gate_args_copy()

            # Readout & reset hooks
            if name == "M":
                # X_ERROR before M models classical readout flip
                if p_readout > 0.0:
                    for t in targs:
                        if t.is_qubit_target:
                            new_circuit.append_operation("X_ERROR", [t], [p_readout])
                new_circuit.append_operation(name, targs, gargs)
                continue
            if name in ("R", "RX", "RY", "RZ"):  # Stim uses 'R' for reset; include axis-specific if present
                new_circuit.append_operation(name, targs, gargs)
                if p_reset > 0.0:
                    for t in targs:
                        if t.is_qubit_target:
                            new_circuit.append_operation("X_ERROR", [t], [p_reset])
                continue

            if name in ONE_Q_GATES:
                new_circuit.append_operation(name, targs, gargs)
 
            # residual 1q around CZ (Pauli channel)
            if p_cz_excess > 0.0:
                s = p_cz_excess / 3.0
                append_pauli_channel_1(a, s, s, s)
                append_pauli_channel_1(b, s, s, s)

        for inst in self.circuit:
            name = inst.name
            targs = inst.targets_copy()
            gate_args = inst.gate_args_copy()

            if name in ONE_Q_GATES:
                # passthrough op
                new_circuit.append_operation(name, targs, gate_args)
                # 1q GPT injection
 
                if twirl_after_1q_gates:
                    for t in targs:
                        if t.is_qubit_target:
                            q = t.value
 
                            seen_qubits.add(q)
                            add_pauli_ch1(q, px, py, pz)
                continue

            if name in TWO_Q_AS_CZ:
                qs = [t.value for t in targs if t.is_qubit_target]
                if len(qs) == 2:
                    current_cz_pairs.append((qs[0], qs[1]))
                    seen_qubits.update(qs)
                new_circuit.append_operation(name, targs, gargs)
                continue

            if name == "TICK":
                # inject all correlated CZ for this window
                for a, b in current_cz_pairs:
                    inject_cz_pair(a, b)
                current_cz_pairs.clear()
                # optional idle twirl + residual idle
                if twirl_idles_each_tick and seen_qubits:
                    ex = p_idle_excess / 3.0
                    for q in sorted(seen_qubits):
                        add_pauli_ch1(q, px + ex, py + ex, pz + ex)
                new_circuit.append_operation(name, targs, gargs)
                continue

            # passthrough any other op (DETECTOR, OBSERVABLE_INCLUDE, MPP, SHIFT_COORDS, etc.)
            new_circuit.append_operation(name, targs, gargs)

        # Flush trailing CZs if circuit doesn't end with TICK
        for a, b in current_cz_pairs:
            inject_cz_pair(a, b)
        self.circuit = new_circuit
 
                            all_qubits_seen.add(q)
                            append_pauli_channel_1(q, pX, pY, pZ)
                continue

            if name in TWO_Q_CZ:
                # record pair inside current TICK window
                qubits = [t.value for t in targs if t.is_qubit_target]
                if len(qubits) == 2:
                    a, b = qubits
                    all_qubits_seen.update(qubits)
                    current_cz_pairs.append((a, b))
                # passthrough op
                new_circuit.append_operation(name, targs, gate_args)
                continue

            if name == "TICK":
                # end of window -> inject correlated CZ errors for all pairs collected
                for (a, b) in current_cz_pairs:
                    inject_cz_correlated(a, b)
                current_cz_pairs.clear()
                # Idle twirl on every seen qubit if requested
                if twirl_idles_each_tick and all_qubits_seen:
                    for q in sorted(all_qubits_seen):
                        append_pauli_channel_1(
                            q,
                            pX + p_idle_excess / 3.0,
                            pY + p_idle_excess / 3.0,
                            pZ + p_idle_excess / 3.0,
                        )
                new_circuit.append_operation(name, targs, gate_args)
                continue

            # passthrough any other operation (DETECTOR, MPP, OBSERVABLE_INCLUDE, coords, etc.)
            new_circuit.append_operation(name, targs, gate_args)

        # Flush a last window if circuit doesn't end with TICK
        for (a, b) in current_cz_pairs:
            inject_cz_correlated(a, b)
        self.circuit = new_circuit
 
    def apply_paper_aligned_noise(self, config):
        """Adjust noise parameters based on a paper-aligned configuration.

        The paper-aligned model specifies detailed physical error rates. For the
        purposes of this simulator we only map a small subset of those
        parameters onto the existing depolarization, leakage, and cross-talk
        knobs. Any parameters not present in ``config`` retain their existing
        values.
        """
        self.depolarization = config.get("p_1q_excess", self.depolarization)
        self.leakage_rate = config.get("p_cz_leak_11_to_02", self.leakage_rate)
        self.cross_talk = config.get("p_cz_crosstalk_ZZ", self.cross_talk)
        # Rebuild the circuit so that updated parameters take effect.
        self.circuit = self._build_noisy_circuit()
 
