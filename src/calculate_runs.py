from typing import Iterable, Optional, Iterable


from pauli_lindblad_per.framework.percircuit import PERCircuit
from pauli_lindblad_per.tomography.processorspec import ProcessorSpec
from pauli_lindblad_per.tomography.layerlearning import LayerLearning
from pauli_lindblad_per.primitives.processor import QiskitProcessor
from pauli_lindblad_per.primitives.circuit import QiskitCircuit

import qiskit


def _wrap_circuit(qc):
    """
    Nimmt einen Qiskit-QuantumCircuit oder bereits einen primitives.Circuit.
    Gibt in beiden Fällen ein primitives.Circuit-Objekt zurück.
    """
    if qc.__class__.__name__ == "QuantumCircuit":
        return QiskitCircuit(qc)
    return qc


def count_tomography_runs(
    *,
    qc,
    backend,
    phys_qubits,
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
        backend: Qiskit-Backend (für QiskitProcessor)
        phys_qubits: Liste der verwendeten phys. Qubits 
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
    subgraph = (len(phys_qubits) != backend.num_qubits)
    processor = QiskitProcessor(backend, subgraph=subgraph)
    procspec = ProcessorSpec(processor, phys_qubits)

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

    res_tomo = {
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
        res_tomo["backend_runs"] = circuits * shots
        
    print("=== Tomography ===")
    print("Circuits:      ", res_tomo["circuits"])
    if "backend_runs" in res_tomo:
        print("Backend runs: ", res_tomo["backend_runs"])

    print("Details:")
    print("  N_layers        :", res_tomo["details"]["num_layers"])
    print("  num_meas_bases  :", res_tomo["details"]["num_meas_bases"])
    print("  depth_count     :", res_tomo["details"]["depth_count"])
    print("  samples         :", res_tomo["details"]["samples"])
    print("  single_samples  :", res_tomo["details"]["single_samples"])
    print("  sum_single_bases:", res_tomo["details"]["sum_single_bases"])

    return res_tomo


# -------- PER -------

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
    num_meas_bases_per = len(qiskit.quantum_info.PauliList(pauli_list).group_commuting(qubit_wise=True))
    num_strengths = sum(1 for _ in noise_strengths)
    circuits = len([qc])*num_meas_bases_per * num_strengths * samples

    res_per = {
        "circuits": circuits,
        "details": {
            "num_meas_bases_per": num_meas_bases_per,
            "num_noise_strengths": num_strengths,
            "samples": samples,
        },
    }
    if shots is not None:
        res_per["backend_runs"] = circuits * shots
        
    # PER Ergebnisse
    print("\n=== PER ===")
    print("Circuits:      ", res_per["circuits"])
    if "backend_runs" in res_per:
        print("Backend runs: ", res_per["backend_runs"])

    print("Details:")
    print("  num_meas_bases_per :", res_per["details"]["num_meas_bases_per"])
    print("  num_noise_strengths:", res_per["details"]["num_noise_strengths"])
    print("  samples            :", res_per["details"]["samples"])
    return res_per