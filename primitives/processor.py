from abc import ABC, abstractmethod
from primitives.circuit import Circuit, QiskitCircuit
from primitives.pauli import QiskitPauli

class Processor(ABC):
    """A wrapper for interacting with a qpu backend. This object is responsible for
    reporting the processor topology and transpiling circuits into the native gate set."""

    @abstractmethod
    def sub_map(self, qubits : int):
        """Return an undirected edge list in the form of tuples of ints representing connections
        between qubits at those hardware addresses"""

    @abstractmethod
    def transpile(self, circuit : Circuit, inst_map, **kwargs):
        """Transpile a circuit into the native gateset"""
    
    @property
    @abstractmethod
    def pauli_type(self):
        """Returns the native Pauli type associated"""


from qiskit import transpile

class QiskitProcessor(Processor):
    """Implementaton of a processor wrapper for the Qiskit API"""

    def __init__(self, backend, subgraph = False):
        self._qpu = backend
        self.subgraph = subgraph

    def sub_map(self, inst_map):
        return self._qpu.coupling_map.graph.subgraph(inst_map)

    #def transpile(self, circuit : QiskitCircuit, inst_map, **kwargs):
        #return QiskitCircuit(transpile(circuits= circuit.qc, backend = self._qpu, **kwargs))
        
    def transpile(self, circuit: QiskitCircuit, used_qubits=None, **kwargs):
        if self.subgraph:
            if used_qubits is None:
                raise ValueError("used_qubits must be provided when subgraph=True")

            from qiskit.transpiler import CouplingMap

            cmap = CouplingMap(couplinglist=[
                (u, v) for u, v in self._qpu.configuration().coupling_map
                if u in used_qubits and v in used_qubits
            ])
            return QiskitCircuit(transpile(
                circuits=circuit.qc,
                backend=self._qpu,
                initial_layout=used_qubits,
                coupling_map=cmap,
                layout_method='trivial',
                **kwargs
            ))
        else:
            return QiskitCircuit(transpile(
                circuits=circuit.qc,
                backend=self._qpu,
                **kwargs
            ))


    @property
    def pauli_type(self):
        return QiskitPauli