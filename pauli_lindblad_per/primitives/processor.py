from abc import ABC, abstractmethod
from pauli_lindblad_per.primitives.circuit import Circuit, QiskitCircuit
from pauli_lindblad_per.primitives.pauli import QiskitPauli
from qiskit.transpiler import CouplingMap
from qiskit import transpile as qiskit_transpile


class Processor(ABC):
    """A wrapper for interacting with a qpu backend. This object is responsible for
    reporting the processor topology and transpiling circuits into the native gate set."""

    @abstractmethod
    def sub_map(self, qubits : int):
        """Return an undirected edge list in the form of tuples of ints representing connections
        between qubits at those hardware addresses"""

    @abstractmethod
    def transpile(self, circuit : Circuit, used_qubits, **kwargs):
        """Transpile a circuit into the native gateset"""
    
    @property
    @abstractmethod
    def pauli_type(self):
        """Returns the native Pauli type associated"""

class QiskitProcessor(Processor):
    """Implementaton of a processor wrapper for the Qiskit API"""

    def __init__(self, backend, subgraph = False):
        self._qpu = backend
        self.subgraph = subgraph

    def sub_map(self, used_qubits):
        return self._qpu.coupling_map.graph.subgraph(used_qubits)
        
    def transpile(self, circuit: QiskitCircuit, used_qubits=None, **kwargs):
        if self.subgraph:
            if used_qubits is None:
                raise ValueError("used_qubits must be provided when subgraph=True")

            # Sub-CouplingMap nur für die ausgewählten Qubits
            cmap = CouplingMap(couplinglist=[
                (u, v)
                for (u, v) in self._qpu.configuration().coupling_map
                if u in used_qubits and v in used_qubits
            ])

            # Transpile mit festem Layout + Sub-CouplingMap
            tqc = qiskit_transpile(
                circuits=circuit.qc,
                coupling_map=cmap,
                initial_layout=used_qubits,    # virtuell 0..n → physisch diese IDs
                optimization_level=0,
                basis_gates=self._qpu.configuration().basis_gates,
                **kwargs
            )
        else:
            tqc = qiskit_transpile(
                circuits=circuit.qc,
                backend=self._qpu,
                optimization_level=0,
                **kwargs
            )

        return QiskitCircuit(tqc)


    @property
    def pauli_type(self):
        return QiskitPauli