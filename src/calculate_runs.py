from typing import Iterable, Optional, Iterable

# --- Imports aus deinem Repo ---
from framework.percircuit import PERCircuit
from tomography.processorspec import ProcessorSpec
from tomography.layerlearning import LayerLearning

# Qiskit-Wrapper (in deinem Repo unter primitives)
from primitives.processor import QiskitProcessor
from primitives.circuit import QiskitCircuit


def _wrap_circuit(qc):
    """
    Nimmt einen Qiskit-QuantumCircuit oder bereits einen primitives.Circuit.
    Gibt in beiden Fällen ein primitives.Circuit-Objekt zurück.
    """
    if qc.__class__.__name__ == "QuantumCircuit":
        return QiskitCircuit(qc)
    return qc


def count_tomography_runs_from_repo(
    *,
    qc,
    inst_map,
    backend,
    used_qubits,
    depths: Iterable[int],
    samples: int,
    single_samples: int,
    shots: Optional[int] = None
):
    """
    Zählt die Anzahl erzeugter Circuits (und optional Backend-Runs) für:
        experiment.generate(samples=..., single_samples=..., depths=[...])
        experiment.run(executor)

    Ermittelt:
      - #Layer:   aus PERCircuit(qc)._layers (eindeutige cliff_layer)
      - Single-Bases je Layer: via LayerLearning._single_bases()
      - num_meas_bases: fest = 9

    Args:
        qc: Qiskit QuantumCircuit oder primitives.Circuit
        inst_map: Mapping der (virtuellen) Qubits
        backend: Qiskit-Backend (für QiskitProcessor)
        used_qubits: Liste der verwendeten phys. Qubits 
        depths: z. B. [2,4,16,32,64]
        samples: Stichproben je (Basis, Tiefe)
        single_samples: Stichproben je Single-Basis
        shots: Optional – falls gesetzt, wird circuits * shots zusätzlich zurückgegeben

    Returns:
        dict mit:
          - "circuits": Anzahl der erzeugten Circuits
          - "backend_runs": (nur wenn shots übergeben) circuits * shots
          - "details": kleine Aufschlüsselung
    """
    # 1) Circuit normalisieren & in PER-Layer zerlegen
    circ_wrap = _wrap_circuit(qc)
    per_circ = PERCircuit(circ_wrap)

    # Menge eindeutiger Clifford-Layer ermitteln
    profiles = set()
    for layer in per_circ._layers:
        try:
            profiles.add(layer.cliff_layer)           # bevorzugt: echte Objekte (haben __eq__/__hash__)
        except TypeError:
            profiles.add(repr(layer.cliff_layer))     # Fallback, falls nicht hashbar

    num_layers = len(profiles)

    # 2) ProcessorSpec aufsetzen (braucht Processor-Wrapper)
    subgraph = (len(used_qubits) != backend.num_qubits)
    processor = QiskitProcessor(backend, subgraph=subgraph)
    procspec = ProcessorSpec(inst_map, processor, used_qubits)

    # 3) Single-Bases pro Layer bestimmen (mit LayerLearning)
    single_bases_counts = []
    for cliff_layer in profiles:
        ll = LayerLearning(cliff_layer, procspec)
        ll._single_bases()  # befüllt ll.single_bases
        single_bases_counts.append(len(ll.single_bases))

    num_meas_bases = 9  # festgelegt
    depth_count = sum(1 for _ in depths)

    multi_part = num_layers * (num_meas_bases * depth_count * samples)
    single_part = sum(single_bases_counts) * single_samples
    circuits = multi_part + single_part

    result = {
        "circuits": circuits,
        "details": {
            "num_layers": num_layers,
            "num_meas_bases": num_meas_bases,
            "depth_count": depth_count,
            "samples": samples,
            "single_samples": single_samples,
            "sum_single_bases": sum(single_bases_counts),
        },
    }
    if shots is not None:
        result["backend_runs"] = circuits * shots
    return result


# -------- PER --------

def _normalize_pauli_label(label: str) -> str:
    s = "".join(label.split()).upper()
    for ch in s:
        if ch not in "IXYZ":
            raise ValueError(f"Ungültiges Pauli-Zeichen: {ch!r}")
    return s

def count_per_meas_bases_from_paulis(pauli_list: Iterable[str]) -> int:
    """
    Minimale Zahl unterschiedlicher Messbasen, die alle Paulis in pauli_list abdecken.
    """
    bases = set()
    for lbl in pauli_list:
        s = _normalize_pauli_label(lbl)
        basis = "".join('Z' if ch == 'I' else ch for ch in s)
        bases.add(basis)
    return len(bases)

def count_per_runs(
    *,
    qc,
    pauli_list: Iterable[str],
    noise_strengths: Iterable[object],
    samples: int,
    shots: Optional[int] = None
):
    """
    Zählt die Circuits (und optional Backend-Runs) für:
        perexp.generate(expectations=pauli_list, samples=..., noise_strengths=[...])
        perexp.run(executor)

    Formel:
      #Circuits = #PER_Messbasen * len(noise_strengths) * samples
    """
    num_meas_bases_per = count_per_meas_bases_from_paulis(pauli_list)
    num_strengths = sum(1 for _ in noise_strengths)
    circuits = len([qc])*num_meas_bases_per * num_strengths * samples

    out = {
        "circuits": circuits,
        "details": {
            "num_meas_bases_per": num_meas_bases_per,
            "num_noise_strengths": num_strengths,
            "samples": samples,
        },
    }
    if shots is not None:
        out["backend_runs"] = circuits * shots
    return out

