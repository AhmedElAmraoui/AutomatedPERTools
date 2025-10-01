import qiskit 

#Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_gate_map, plot_error_map
from qiskit.circuit import QuantumRegister
from qiskit.quantum_info import Pauli, pauli_basis, SuperOp, PTM, Operator
#from qiskit.providers.aer.noise import NoiseModel, pauli_error
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
#from qiskit.providers.fake_provider import FakeVigoV2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import Aer, AerSimulator,QasmSimulator
from random import random
from qiskit.transpiler import CouplingMap

#python imports
from random import choice, choices
from itertools import product, permutations, cycle
from scipy.optimize import curve_fit, nnls
from matplotlib import pyplot as plt
import numpy as np
from qiskit.quantum_info import Pauli
from matplotlib.patches import Rectangle
from itertools import product
from matplotlib.patches import Rectangle
from itertools import combinations

plt.style.use("ggplot")

"""def meas_bases(NUM_BASES, n, connectivity):
    bases = [['I']*n for i in range(NUM_BASES)]

    for vertex in range(n):

        children = connectivity.neighbors(vertex)
        predecessors = [c for c in children if c < vertex]

        match len(predecessors):
            #trivial if no predecessors
            case 0:            
                cycp = cycle("XYZ")
                
                for i,_ in enumerate(bases):
                    bases[i][vertex] = next(cycp)

            #Choose p1:"XXXYYYZZZ" and p2:"XYZXYZXYZ" if one predecessor
            case 1:
                pred, = predecessors
                #store permutation of indices so that predecessor has X,X,X,Y,Y,Y,Z,Z,Z
                _,bases = list(zip(*sorted(zip([p[pred] for p in bases], bases))))
                cycp = cycle("XYZ")

                for i,_ in enumerate(bases):
                    bases[i][vertex] = next(cycp)

            case 2:
                pred0,pred1 = sorted(predecessors)
                #store permutation of indices so that first predecessor has X,X,X,Y,Y,Y,Z,Z,Z
                _,bases = list(zip(*sorted(zip([p[pred0] for p in bases], bases))))
                #ordering of the vertex with two predecessor has X,Y,Z,Z,X,Y,Y,Z,X
                order =  "XYZZXYYZX"
                for i,_ in enumerate(bases):             
                    bases[i][vertex] = order[i]
                    
            case _: #processor needs to have connectivity so that there are <= 2 predecessors
                raise Exception("Three or more predecessors encountered")
    # the pauli basis in 0,1,2,3 indices are rearranged according the qiskit 3,2,1,0 indices notation.
    bases = [Pauli("".join(string[::-1])) for string in bases]

    return bases"""


def meas_bases(NUM_BASES, n, connectivity, qubit_subset=None):
    """
    Generiert Pauli-Messbasen für n Qubits, beschränkt auf die Qubits in qubit_subset.
    Alle anderen Qubits erhalten 'I' (keine Messung).

    Args:
        NUM_BASES (int): Anzahl der globalen Basen (z.B. 9).
        n (int): Gesamtzahl der Qubits.
        connectivity: Graph-Objekt mit .neighbors(v) für jede Kante.
        qubit_subset (Iterable[int] oder None): Indizes der Qubits, die tatsächlich gemessen werden.
            Wenn None, wird ganz {0,...,n-1} verwendet.

    Returns:
        List[Pauli]: Liste von NUM_BASES Pauli-Strings der Länge n.
    """
    # Standard-Subset: alle Qubits
    if qubit_subset is None:
        qubit_subset = set(range(n))
    else:
        qubit_subset = set(qubit_subset)

    # Starte alle Basen mit 'I'
    bases = [['I'] * n for _ in range(NUM_BASES)]

    # Iteriere nur über die zu messenden Qubits, in aufsteigender Reihenfolge
    for v in sorted(qubit_subset):
        # Finde bereits bearbeitete Vorgänger, aber nur in qubit_subset
        preds = [u for u in connectivity.neighbors(v) if u < v and u in qubit_subset]

        match len(preds):
            case 0:
                # Einfach jede Basis zyklisch mit X,Y,Z füllen
                cycp = cycle("XYZ")
                for row in bases:
                    row[v] = next(cycp)

            case 1:
                # Vorgänger p hat in den current bases jeweils XXXYYYZZZ
                p, = preds
                # Sortiere bases so, dass p an Position X,X,X,Y,Y,Y,Z,Z,Z steht
                _, bases = zip(
                    *sorted(
                        zip([row[p] for row in bases], bases),
                        key=lambda x: x[0]
                    )
                )
                bases = list(map(list, bases))
                # Fülle v zyklisch
                cycp = cycle("XYZ")
                for row in bases:
                    row[v] = next(cycp)

            case 2:
                p0, p1 = sorted(preds)
                # Sortiere nach p0 wie oben
                _, bases = zip(
                    *sorted(
                        zip([row[p0] for row in bases], bases),
                        key=lambda x: x[0]
                    )
                )
                bases = list(map(list, bases))
                # Feste Reihenfolge für p1: "XYZZXYYZX"
                order = "XYZZXYYZX"
                for i, row in enumerate(bases):
                    row[v] = order[i]

            case _:
                raise ValueError(f"Qubit {v} hat ≥3 Vorgänger in der Submenge – Topologie nicht unterstützt")

    # Wandle in Qiskit-Pauli-Objekte um (Achtung: Reihenfolge umdrehen für qiskit-Konvention)
    pauli_bases = [
        Pauli("".join(reversed("".join(row))))
        for row in bases
    ]
    return pauli_bases




#remove the phase from a Pauli
def nophase(pauli):
    return Pauli((pauli.z, pauli.x))

def conjugate(pauli, layer):
    """It gives the Pdagger for noise twirling"""
    return nophase(pauli.evolve(layer))

def instance(
    n, layer, backend, inst_map,
    prep_basis : Pauli, 
    meas_basis : Pauli, 
    noise_repetitions : int, 
    transpiled=True):

    circ = QuantumCircuit(n) #storing the final circuit

    #get preparation ops from desired basis 
    def prep(basis, qubit, qc):
        if basis.equiv(Pauli("X")):
            qc.h(qubit) 
        elif basis.equiv(Pauli("Y")):
            qc.h(qubit)
            qc.s(qubit)

    #apply operators to a quantum circuit to measure in desired pauli basis
    def meas(basis, qubit, qc):
        if basis.equiv(Pauli("X")):
            qc.h(qubit)
        elif basis.equiv(Pauli("Y")):
            qc.sdg(qubit)
            qc.h(qubit)


    pauli_frame = Pauli("I"*n)

    #apply the prep operators to the circuit
    for q,b in enumerate(prep_basis):
        prep(b,q,circ)

    #apply repetitions of noise, including basis-change gates when needed
    for i in range(noise_repetitions):
        circ = circ.compose(layer)
        circ.barrier()


    for q,b in enumerate(meas_basis):        
        meas(b, q, circ)

    circ.measure_all()

    if transpiled:
        circ = transpile(circ, backend, initial_layout=inst_map, optimization_level = 1)

    circ.metadata = {
        "prep_basis":prep_basis,
            "meas_basis":meas_basis, 
            "depth":noise_repetitions
            }

    return circ 

"""def get_model_terms(n, connectivity):
    model_terms = set()
    identity = Pauli("I"*n)    
    #get all weight-two paulis on with suport on nieghboring qubits
    for q1,q2 in connectivity.edge_list():
            for p1, p2 in pauli_basis(2, True):
                pauli = identity.copy()
                pauli[q1] = p1
                pauli[q2] = p2
                model_terms.add(pauli)

    model_terms.remove(identity)
    model_terms = (list(model_terms))
    print("Model terms:", np.sort([m.to_label()[::-1] for m in model_terms]),"\n")
    print(f"Number of model terms: {len(model_terms)}")
    return model_terms"""
    
import numpy as np
from qiskit.quantum_info import Pauli
from itertools import product

def get_model_terms(n, connectivity, qubit_subset=None):
    """
    Ermittelt alle Gewicht-2 Pauli-Terme auf dem Subgraphen, der durch qubit_subset gegeben ist.

    Args:
        n (int):
            Gesamtzahl der Qubits im System.
        connectivity:
            Graph-Objekt mit Methode .edge_list(), die Kanten als Tuples (q1, q2) liefert.
        qubit_subset (Iterable[int] oder None):
            Indizes der Qubits, die zum Subgraphen gehören. Wenn None, wird der
            gesamte Graph verwendet.

    Returns:
        List[Pauli]:
            Alle gewicht-2 Pauli-Operatoren (ohne die Identität), deren Stützstellen in qubit_subset liegen.
    """
    # Definiere den Subgraphen als Menge
    if qubit_subset is None:
        qubit_subset = set(range(n))
    else:
        qubit_subset = set(qubit_subset)

    # Leere Identität auf n Qubits
    identity = Pauli("I" * n)

    model_terms = set()
    # Pauli-Basis auf zwei Qubits (aus qiskit.tools oder eigenem Helper)
    two_qubit_paulis = list(product([Pauli("I"), Pauli("X"), Pauli("Y"), Pauli("Z")], repeat=2))
    # Wir wollen nur die gewicht-2 Terme, also (p1,p2) != (I,I):
    two_qubit_paulis = [(p1, p2) for p1, p2 in two_qubit_paulis if not (p1.to_label()=="I" and p2.to_label()=="I")]

    # Gehe alle Kanten durch, aber nur innerhalb qubit_subset
    for q1, q2 in connectivity.edge_list():
        if q1 in qubit_subset and q2 in qubit_subset:
            for p1, p2 in two_qubit_paulis:
                pauli = identity.copy()
                pauli[q1] = p1
                pauli[q2] = p2
                model_terms.add(pauli)

    # Entferne versehentliche Identität (falls vorhanden)
    model_terms.discard(identity)

    model_terms = list(model_terms)
    labels = np.sort([m.to_label()[::-1] for m in model_terms])
    print("Model terms:", labels, "\n")
    print(f"Number of model terms: {len(model_terms)}")
    return model_terms

import numpy as np
from qiskit.quantum_info import SuperOp, Operator
from functools import reduce

def embed_channel(local_channel: SuperOp,
                  acting_qubits: list[int],
                  total_qubits: int) -> SuperOp:
    """
    Embeds an m-qubit SuperOp into an N-qubit identity by tensoring:
        I ⊗ ... ⊗ local_channel ⊗ ... ⊗ I

    Args:
        local_channel: SuperOp acting on m qubits.
        acting_qubits: List of m distinct qubit indices (0 <= i < N).
        total_qubits: Gesamtzahl der Qubits N.

    Returns:
        global_channel: SuperOp auf N Qubits.
    """
    # Sanity-Check: m = # acting_qubits
    m = len(acting_qubits)
    # qiskit SuperOp has .input_dims() == [2,2,...] für m Qubits
    in_dims, _ = local_channel.dims
    if len(in_dims) != m:
        raise ValueError(f"Local channel hat {len(in_dims)} Qubits, "
                         f"acting_qubits liefert aber m={m}")

    # Identity-SuperOp auf 1 Qubit
    id_op = SuperOp(Operator(np.eye(2)))

    # Wir brauchen nur _eine_ Einfügung von local_channel,
    # alle weiteren acting_qubits-Plätze werden übersprungen.
    acting_set = set(acting_qubits)
    inserted = False
    ops = []

    for i in range(total_qubits):
        if i in acting_set:
            if not inserted:
                ops.append(local_channel)
                inserted = True
            else:
                # skip: ist schon durch local_channel abgedeckt
                continue
        else:
            ops.append(id_op)

    # Jetzt tensorproduktweise zusammenschieben (right-to-left)
    # reduce mit .expand entspricht op_n ⊗ ... ⊗ op_2 ⊗ op_1
    global_channel = reduce(
        lambda A, B: A.expand(B),
        reversed(ops)
    )
    return global_channel



def plot_fidelities_grouped_by_qubit(model_terms, measured_coeffs, ideal_coeffs, coupling_list, title="Fidelity Comparison", ylabel= "Coefficients"):
    """
    Plots measured vs ideal fidelities grouped by qubit or qubit pair,
    sorted alphabetically within groups, with clean Pauli labels and separators.
    """
    n_qubits = len(model_terms[0].to_label())
    allowed_pairs = {(min(a, b), max(a, b)) for (a, b) in coupling_list}

    block_colors = ['#d0e1f9', '#f9d0d0', '#d0f9d9', '#f9f5d0', '#e0d0f9', '#f0c0f9']

    def pauli_support(pauli):
        return [i for i, p in enumerate(pauli.to_label()) if p != 'I']

    def compact_pauli_label(pauli):
        return ''.join(p for p in pauli.to_label() if p != 'I')

    def pauli_key_string(pauli):
        return compact_pauli_label(pauli)

    # Gruppieren nach Gewicht und beteiligten Qubits
    groups = {}
    for i, term in enumerate(model_terms):
        support = pauli_support(term)
        weight = len(support)
        if weight == 1:
            key = (1, support[0])
        elif weight == 2:
            pair = tuple(sorted(support))
            if pair not in allowed_pairs:
                continue
            key = (2, pair)
        else:
            continue
        groups.setdefault(key, []).append(i)

    # Sortierte Keys: erst Gewicht 1, dann 2, jeweils nach Qubit-Index/Paar
    sorted_keys = sorted(groups.keys(), key=lambda k: (k[0], k[1]))

    # Plot-Vorbereitung
    fig, ax = plt.subplots(figsize=(14, 6))
    bar_positions = []
    bar_labels = []
    bar_measured = []
    bar_ideal = []
    background_regions = []
    separator_lines = []
    current_index = 0

    for color_index, key in enumerate(sorted_keys):
        indices = groups[key]
        color = block_colors[color_index % len(block_colors)]

        # Sortiere innerhalb des Blocks alphabetisch nach Pauli-Kürzel
        sorted_indices = sorted(
            indices,
            key=lambda i: pauli_key_string(model_terms[i])
        )

        background_regions.append((current_index, len(sorted_indices), color))

        for idx in sorted_indices:
            bar_positions.append(current_index)
            bar_labels.append(compact_pauli_label(model_terms[idx]))
            bar_measured.append(measured_coeffs[idx])
            bar_ideal.append(ideal_coeffs[idx])
            current_index += 1

        separator_lines.append(current_index - 0.5)

    # Balken plotten
    bar_positions = np.array(bar_positions)
    ax.bar(bar_positions + 0.2, bar_measured, 0.4, color='tab:blue', label='Measured')
    ax.bar(bar_positions - 0.2, bar_ideal, 0.4, color='tab:green', label='Ideal')

    # Hintergrundfarben + Beschriftung oben
    for start, length, color in background_regions:
        ax.add_patch(Rectangle((start - 0.5, ax.get_ylim()[0]), length, ax.get_ylim()[1] - ax.get_ylim()[0],
                               color=color, alpha=0.4, zorder=0))
        key = sorted_keys[background_regions.index((start, length, color))]
        label = f"Qu {key[1]}" if key[0] == 1 else f"Qu pair {key[1]}"
        plt.text(start + length / 2 - 0.5, ax.get_ylim()[1] * 0.95, label,
                 ha='center', va='top', fontsize=9, weight='bold', color='black')

    # Schwarze Trennlinien zwischen Gruppen
    for x in separator_lines[:-1]:
        ax.axvline(x=x, color='black', linestyle='-', linewidth=1)

    # Achsen & Formatierung
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, rotation=90)
    ax.set_title(title)
    ax.set_xlabel("Support of model terms")
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    plt.show()
    

#graph a subset of the measured expectation values and plot fits
def graph(*paulis, basis_dict, expfit, depths):
    colcy = cycle(["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:cyan", "tab:brown", "tab:pink", "tab:gray", "tab:olive"])
    for p in paulis:
        c = next(colcy)
        data = basis_dict[p]['expectation']
        popt, pcov = curve_fit(expfit, depths, data, p0=[.9,.01])
        xrange = np.linspace(0,np.max(depths))
        plt.plot(xrange, [expfit(x, *popt) for x in xrange], color=c)
        plt.plot(depths, data, color = c, marker="o", linestyle = 'None')
    plt.title("Expectation vs Depth")
    plt.xlabel("Depth")
    plt.ylabel("Fidelity")
    
    
def check_fidelity_discrepancies(model_terms,layer, basis_dict, ideal_fidelities, meas_fidelities, threshold=0.01):
    print(f"{'Pauli':<10} {'Measured':<10} {'Ideal':<10} {'Δ':<8} {'Type':<6} {'Conj':<6} {'Conj ∈ model_terms'}")
    print("-"*60)
    for i in range(len(model_terms)):
        p = model_terms[i]
        fid_meas = np.real(meas_fidelities[i])
        fid_ideal = np.real(ideal_fidelities[i])
        typ = basis_dict.get(p, {}).get("type", "?")

        delta = abs(fid_meas - fid_ideal)/fid_ideal
        if delta > threshold:
            conj_in_model = conjugate(p, layer) in model_terms
            print(f"{p.to_label():<10} {fid_meas:<10.3f} {fid_ideal:<10.3f} {delta:<8.3f} {typ:<6} {conjugate(p, layer)} {conj_in_model}")
            
    

def compute_pauli_fidelity_4q(pauli_label, error_pauli='XX', error_prob=0.02, target_qubits=[1,2]):
    """
    Berechnet f_P für ein gegebenes 4-Qubit-Pauli unter einem 2-Qubit-PauliError
    """
    p = Pauli(pauli_label[::-1])  # little endian
    p_label = list(p.to_label())

    # Extrahiere betroffene Teilpaulis
    reduced = ''.join([p_label[q] for q in target_qubits])
    reduced_p = Pauli(reduced)

    # Fehler: E P E = E P E (Pauli-Wirkung)
    e = Pauli(error_pauli)
    transformed = e @ reduced_p @ e

    # Eigenwert ermitteln: ±1
    if transformed == reduced_p:
        factor = 1
    elif transformed == -reduced_p:
        factor = -1
    else:
        return 0.0  # anderer Pauli-Typ → keine Überlappung → kein f_P

    # Gewichtung: p * factor + (1-p)
    fidelity = error_prob * factor + (1 - error_prob)
    return fidelity.real



def pauli_basis_2q():
    return [Pauli(p1 + p2) for p1, p2 in product('IXYZ', repeat=2)]

def superop_to_ptm(sup: SuperOp) -> np.ndarray:
    M = sup.data
    d2 = M.shape[0]
    d  = int(np.sqrt(d2))
    m  = int(np.log2(d))
    # Orthornormierte Pauli-Vektoren als Spalten
    vecs = [P.to_matrix().reshape(d*d,) / np.sqrt(d)
            for P in (Pauli(''.join(lbl)) for lbl in product('IXYZ', repeat=m))]
    Q = np.column_stack(vecs)
    return Q.conj().T @ M @ Q

def compute_local_ptm_diagonals(error_specs):
    basis2 = pauli_basis_2q()
    local_ptms = {}
    for pair, ops_probs in error_specs.items():
        # Ergänze Rest‐Term (II)
        p_rest = 1 - sum(p for _, p in ops_probs)
        plist = ops_probs + [(Pauli("II"), p_rest)]
        sup = SuperOp(pauli_error(plist).to_quantumchannel())
        R = superop_to_ptm(sup)
        diag = np.diag(R)
        # Map Label → f_local
        ptm_dict = {basis2[i].to_label(): float(diag[i])
                    for i in range(len(basis2))}
        local_ptms[pair] = ptm_dict
    return local_ptms

def get_model_fidelities(n, cx_pairs, error_specs, model_terms):
    """
    Berechnet f_j nur für genau die Pauli-Labels in model_terms.

    Args:
        n (int): Gesamtzahl der Qubits.
        cx_pairs (list of tuple): Liste der fehlerbehafteten Zwei-Qubit-Paare.
        local_ptms (dict): Mapping pair -> {2-Qubit-Pauli-Label -> f_local}.
        model_terms (list of str or Pauli): Länge-n Pauli-Strings (nur Gewicht-2, topologisch).

    Returns:
        dict: Mapping aus String-Label → fidelity f_j.
    """
    local_ptms = compute_local_ptm_diagonals(error_specs)
    
    def to_label(lbl):
        # Konvertiere Pauli-Objekt oder numpy‐String o.Ä. in echtes Python-str
        if isinstance(lbl, Pauli):
            s = lbl.to_label()
        else:
            s = str(lbl)
        return s

    def find_pair(i):
        for pair in cx_pairs:
            if i in pair:
                return pair
        return None

    out = {}
    for raw in model_terms:
        lbl = to_label(raw)
        if len(lbl) != n:
            raise ValueError(f"Term '{lbl}' hat nicht die richtige Länge {n}")
        # Support = Indizes, an denen wirklich X, Y oder Z steht
        support = [i for i, c in enumerate(lbl) if c in 'XYZ']
        w = len(support)

        if w == 0:
            # nur Identity → fidelity = 1
            f = 1.0

        elif w == 1:
            # Einzelpauli
            i = support[0]
            p = lbl[i]
            pair = find_pair(i)
            if pair:
                a, b = sorted(pair)
                # Lokales Label
                loc = (p + 'I') if i == a else ('I' + p)
                f = local_ptms[pair].get(loc, 1.0)
            else:
                f = 1.0

        elif w == 2:
            # Zwei-Pauli, topologisch per model_terms vorgegeben
            i, j = support
            pi, pj = lbl[i], lbl[j]
            pair_i = find_pair(i)
            pair_j = find_pair(j)

            if pair_i and pair_i == pair_j:
                # beide im selben CX-Paar
                a, b = sorted(pair_i)
                if (i, j) == (a, b):
                    loc = pi + pj
                else:
                    loc = pj + pi
                f = local_ptms[pair_i].get(loc, 1.0)
            else:
                # unabhängige Beiträge multiplizieren
                f = 1.0
                if pair_i:
                    a, b = sorted(pair_i)
                    loc = (pi + 'I') if i == a else ('I' + pi)
                    f *= local_ptms[pair_i].get(loc, 1.0)
                if pair_j:
                    a, b = sorted(pair_j)
                    loc = (pj + 'I') if j == a else ('I' + pj)
                    f *= local_ptms[pair_j].get(loc, 1.0)

        else:
            # Gewichte > 2 nicht erwartet
            raise ValueError(f"Term '{lbl}' hat Hamming-Gewicht {w} > 2")

        out[lbl] = f

    return out

def plot_weight1_weight2_fidelities(fidelities, n, coupling_list,
                                    title="Fidelity Comparison",
                                    ylabel="Fidelity"):
    """
    Plots fidelities for weight-1 and weight-2 Pauli terms,
    grouped by qubit (weight-1) and qubit pair (weight-2).

    Args:
        fidelities (dict): Mapping from full-length Pauli-label (str) to fidelity f.
        n (int): Total number of qubits.
        coupling_list (list of tuple): List of allowed qubit couplings (pairs).
        title (str): Plot title.
        ylabel (str): Label for the y-axis.
    """
    allowed_pairs = {tuple(sorted(pair)) for pair in coupling_list}

    # Build lists of labels and fidelities for weight-1 and weight-2
    labels = []
    measured = []
    ideal = []
    for label, f in fidelities.items():
        weight = n - label.count('I')
        if weight == 1:
            labels.append(label)
            measured.append(f)
            ideal.append(1.0)
        elif weight == 2:
            support = tuple(sorted(i for i, ch in enumerate(label) if ch != 'I'))
            if support in allowed_pairs:
                labels.append(label)
                measured.append(f)
                ideal.append(1.0)

    # Group indices by (weight, qubit or pair)
    groups = {}
    for idx, label in enumerate(labels):
        support = [i for i, ch in enumerate(label) if ch != 'I']
        if len(support) == 1:
            key = (1, support[0])
        else:
            key = (2, tuple(sorted(support)))
        groups.setdefault(key, []).append(idx)

    # Sort keys by weight then by qubit index/pair
    sorted_keys = sorted(groups.keys(), key=lambda k: (k[0], k[1]))

    # Prepare for plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    bar_positions = []
    bar_labels = []
    measured_vals = []
    ideal_vals = []
    bg_regions = []
    separators = []
    current = 0
    colors = ['#d0e1f9', '#f9d0d0', '#d0f9d9', '#f9f5d0', '#e0d0f9', '#f0c0f9']

    for color_idx, key in enumerate(sorted_keys):
        idxs = sorted(groups[key], key=lambda i: labels[i].replace('I', ''))
        start = current
        for i in idxs:
            bar_positions.append(current)
            bar_labels.append(''.join(ch for ch in labels[i] if ch != 'I'))
            measured_vals.append(1-measured[i])
            ideal_vals.append(1-ideal[i])
            current += 1
        bg_regions.append((start, len(idxs), colors[color_idx % len(colors)]))
        separators.append(current - 0.5)

    # Plot bars
    bar_positions = np.array(bar_positions)
    ax.bar(bar_positions + 0.2, measured_vals, 0.4, label='Measured', color='tab:blue')
    ax.bar(bar_positions - 0.2, ideal_vals,    0.4, label='Ideal',    color='tab:green')

    # Add background and group labels
    y0, y1 = ax.get_ylim()
    for (start, length, color), key in zip(bg_regions, sorted_keys):
        rect = Rectangle((start - 0.5, y0), length, y1 - y0,
                         color=color, alpha=0.4, zorder=0)
        ax.add_patch(rect)
        label = f"Qu{key[1]}" if key[0] == 1 else f"Pair {key[1]}"
        ax.text(start + length/2 - 0.5, y1 * 0.95,
                label, ha='center', va='top', fontsize=9, weight='bold')

    # Draw separator lines
    for s in separators[:-1]:
        ax.axvline(s, color='black', linewidth=1)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, rotation=90)
    ax.set_title(title)
    ax.set_xlabel("Support of Pauli terms")
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    
def get_basis_dict(n, inst_map, layer,backend, model_terms, bases, noise_model):
    
    # QASM simulator is considered, we will add the custom noise model to it for simulations
    sim = QasmSimulator()

    SINGLE = 1
    shots = 1000
    for term in model_terms:
        circ = instance(n, layer, backend, inst_map, prep_basis=conjugate(term, layer), meas_basis=term,noise_repetitions= SINGLE,transpiled=False)
        result_single_counts = sim.run(circ,noise_model=noise_model, shots = shots).result().get_counts()
        
    def get_expectation(pauli, **metadata):
        estimator = 0
        counts = metadata['counts']
        #rostring = metadata['rostring']
        #compute locations of non-idetity terms (reversed indexing)
        pz = list(reversed([{Pauli("I"):'0'}.get(p,'1') for p in pauli]))
        #compute estimator
        for key in counts.keys():
            k = [i for i in key]            
            #compute the overlap in the computational basis
            sgn = sum([{('1','1'):1}.get((pauli_bit, key_bit), 0) for pauli_bit, key_bit in zip(pz, k)])
            #update estimator
            estimator += (-1)**sgn*counts[key]

        return estimator/sum(counts.values())

    def weight(pauli):
        return len([p for p in pauli if not p==Pauli("I")])

    #return True if Paulis differ by "I"s only
    def disjoint(pauli1, pauli2):
        return all([p1==p2 or (p1 == Pauli("I") or p2 == Pauli("I")) for p1,p2 in zip(pauli1, pauli2)])

    #return True if pauli requires a degeracy lifting measurement based on the conditions described above
    def is_single(pauli, layer):
        pair = conjugate(pauli, layer)
        return (pauli in model_terms and pair in model_terms) and pauli != pair


    #find disjoint operators that can be measured simultaneously to find six bases
    pairs = set([frozenset([p,conjugate(p, layer)]) for p in model_terms if is_single(p, layer)])
    #print(pairs)
    single_bases = []
    for p1,p2 in pairs:
        
        for i,pauli in enumerate(single_bases):
            if disjoint(pauli, p1) and disjoint(pauli, p2):
                single_bases[i] = nophase(pauli.compose(p2))
                break
        else:
            if weight(p1)<=weight(p2):
                single_bases.append(p2)
            else:
                single_bases.append(p1)
                
    for p in model_terms:
        pair = conjugate(p,layer)
        if pair not in model_terms:
            single_bases.append(p)    

    #print("bases for singles: ",single_bases)
    
    SINGLE = 1
    circuits = []
    depths = [2,4,8,16,32,64]
    samples = [100]*len(depths)
    #print(samples)
    single_samples = 250
    total = len(bases)*sum(samples)+len(single_bases)*single_samples

    j=0
    for basis, (d,s) in product(bases, zip(depths,samples)):
        for i in range(s):
            circ = instance(n, layer, backend, inst_map,basis, basis, d,transpiled=True)
            circ.metadata["type"] = "double"
            circuits.append(circ)

            j+=1
            print(j,"/",total, end='\r')

    for basis, s in product(single_bases, range(single_samples)):
        circ = instance(n, layer, backend, inst_map,conjugate(basis, layer),basis,SINGLE,transpiled=True)
        circ.metadata["type"] = "single"
        circuits.append(circ)

        j+=1
        print(j,"/",total, end='\r')

    #print(len(circuits))
    
    results = sim.run(circuits, shots=1000, noise_model = noise_model).result().get_counts()

    #Shows whether two pauli operators can be measured simultaneously
    def simultaneous(pauli1, pauli2):
        return all([p1==p2 or p2 == Pauli("I") for p1,p2 in zip(pauli1, pauli2)])

    #Gives a list of all terms in the model that can be measured simultaneously with pauli
    def sim_meas(pauli):
        return [term for term in model_terms if simultaneous(pauli, term)]

    #Gives a list of all terms, in the sparse model or not, that can be measured simultaneously
    #This is used to overdeteremine the model, but since it grows as 2^n, this method can always be
    #replaced with sim_meas
    def all_sim_meas(pauli):
        return [Pauli("".join(p)) for p in product(*zip(pauli.to_label(), "I"*n))]
    #all_sim_meas = sim_meas
    
    #Sort into single and double measurements
    for res,circ in zip(results, circuits):
        circ.metadata["counts"] = res

    singles = []
    doubles = []
    for circ in circuits:
        datum = circ.metadata
        type = datum["type"]
        datum.pop("type")
        if type == "single":
            singles.append(datum)
        elif type == "double":
            doubles.append(datum)
            
    #reorder by measurement basis
    basis_dict = {}
    #improve execution time by storing runs of all_sim_meas for each basis
    sim_measurements = {}
    for datum in doubles:
        #print(datum)
        #get run data
        basis = datum['prep_basis']
        depth = datum['depth']
        #find simultaneous measurements
        if not basis in sim_measurements:
            sim_measurements[basis] = all_sim_meas(basis)
        #aggregate expectation value data for each simultaneous measurement
        for pauli in sim_measurements[basis]:

            expectation = get_expectation(pauli, **datum)
            #set up dictionary
            if not pauli in basis_dict:
                basis_dict[pauli] = {"expectation":[0 for d in depths], "total":[0 for d in depths]}
            
            pair = conjugate(pauli, layer)
            
            #add expectation value to result at depth
            basis_dict[pauli]["expectation"][depths.index(depth)] += expectation
            basis_dict[pauli]["total"][depths.index(depth)] += 1

    for p in model_terms:
        pair = conjugate(p, layer)
        if pair not in model_terms and pair not in basis_dict.keys():
            basis_dict[pair] = {}


    expfit = lambda x,a,b : a*np.exp(-x*b)
    #for each of the simultaneous measurements
    for key in basis_dict.keys():
        for i,d in enumerate(depths):
            #divide by total
            if "expectation" in basis_dict[key]:
                basis_dict[key]["expectation"][i] /= basis_dict[key]["total"][i]
        #try finding exponential fit, default to ideal if no fit found
        try:
            popt, pcov = curve_fit(expfit, depths, basis_dict[key]["expectation"], p0=[.9,.01])
        except:
            popt = 1,0

        #store fidelity and SPAM coefficients
        fidelity = min(1.0, max(0.0, expfit(1,*popt)))
        spam = min(1.0, max(0.0, popt[0]))

        basis_dict[key]["fidelity"] = fidelity
        basis_dict[key]["SPAM"] = spam
        
        pair = conjugate(key, layer)
        if pair not in model_terms and pair in basis_dict.keys():
            basis_dict[pair]["fidelity"] = fidelity
            basis_dict[pair]["SPAM"] = spam
            basis_dict[pair]["type"] = "pair"

        #record whether measurement appears as a pair or as a single fidelity
        if key != pair:
            basis_dict[key]["type"] = "pair"
        else:
            basis_dict[key]["type"] = "single"
            
        singles_dict = {} #store results of single measurements
    sim_measurements = {}
    for datum in singles:
        meas_basis = datum['meas_basis']
        prep_basis = datum['prep_basis']
        #find terms that can be measured simultaneously
        if not meas_basis in sim_measurements:
            sim_measurements[meas_basis] = []
            for term in model_terms:
                if simultaneous(meas_basis, term) and simultaneous(prep_basis, conjugate(term, layer)) and term in single_bases:
                    sim_measurements[meas_basis].append(term)
        #aggregate data together
        for meas in sim_measurements[meas_basis]:
            expectation = get_expectation(meas, **datum)
            #the measurement basis SPAM coefficients are closer because the readout noise, combined
            #with the noise from the last layer, is greater than the state preparation noise
            #print( meas, ', ', np.abs(expectation), ', ', basis_dict[meas]["SPAM"])
            fidelity = np.min([1.0,np.abs(expectation)/basis_dict[meas]["SPAM"]])
            #singles_dict[meas] += fidelity/single_samples
            
            if meas not in singles_dict:
                singles_dict[meas] = []

            singles_dict[meas].append(fidelity)

    #add singles data to basis_dict
    for key in singles_dict:
        avg_fidelity = np.mean(singles_dict[key])
        basis_dict[key]['fidelity'] = avg_fidelity
        basis_dict[key]['type'] = "single"
        
    return basis_dict