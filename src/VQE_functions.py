import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli
from qiskit import QuantumCircuit, transpile
from scipy.optimize import minimize
from qiskit.circuit import ParameterVector
import networkx as nx
from qiskit.primitives import BackendSamplerV2
from scipy.linalg import eigh
from tqdm.auto import tqdm
from qiskit.transpiler import CouplingMap

def get_sites_and_bonds(sites, tol=1e-8):
    """
    Bestimmt die Anzahl der Qubits (Sites) und Bonds (nächste Nachbarn) 
    für eine gegebene Topologie von Sites.

    Die nächsten Nachbarn sind diejenigen Paare von Sites, 
    die den kleinsten Abstand a > 0 voneinander haben.

    Args:
        sites (np.ndarray): Array der Form (N, d), die Koordinaten der Sites.
        tol (float): numerische Toleranz für Abstand-Vergleiche.

    Returns:
        num_qubits (int): Anzahl der Sites.
        bonds (list of tuples): Liste der Bonds (i, j).
    """
    
    if sites is None:
        raise ValueError("Es muss eine Topologie (sites) vorgegeben werden!")
    
    num_qubits = len(sites)
    dists = []
    
    if num_qubits == 0:
        raise ValueError("Die Topologie darf nicht leer sein!")

    # Alle paarweisen Abstände berechnen
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            dist = np.linalg.norm(sites[i] - sites[j])
            dists.append(dist)

    # minimalen Abstand a > 0 finden
    a = min([d for d in dists if d > tol])

    bonds = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            dist = np.linalg.norm(sites[i] - sites[j])
            if abs(dist - a) < tol:  # nur Bonds mit minimalem Abstand
                bonds.append((i, j))

    return num_qubits, bonds


def MFIM_Hamiltonian(J=None, hx=None, hz=None, topology=None, return_paulis=False):
    """
    Baut den Hamiltonian für das Transversale Ising-Modell (MFIM) 
    auf einer gegebenen Topologie.

    Args:
        J (float): Kopplungskonstante (ZZ-Wechselwirkung).
        hx (float): Stärke des X-Feldes.
        hz (float): Stärke des Z-Feldes.
        topology (np.ndarray): Array der Form (N, d), Koordinaten der Sites.
        return_paulis (bool): Falls True, zusätzlich die Pauli-Strings zurückgeben.

    Returns:
        SparsePauliOp oder (SparsePauliOp, list[str])
    """
    if topology is None:
        raise ValueError("Es muss eine Topologie (topology) vorgegeben werden (\u00B11)!")
    if J is None:
        raise ValueError("Parameter J (Wechselwirkung) muss gesetzt sein!")
    if hx is None:
        raise ValueError("Parameter hx (X-Feld) muss gesetzt sein, kann auch null sein!")
    if hz is None:
        raise ValueError("Parameter hz (Z-Feld) muss gesetzt sein, kann auch null sein!!")

    num_qubits, bonds = get_sites_and_bonds(topology)

    paulis = []
    coeffs = []

    # ZZ-Terme
    for i, j in bonds:
        z_str = ['I'] * num_qubits
        z_str[i] = 'Z'
        z_str[j] = 'Z'
        pauli = ''.join(reversed(z_str))
        paulis.append(pauli)
        coeffs.append(J)

    # X- und Z-Feld-Terme
    for i in range(num_qubits):
        x_str = ['I'] * num_qubits
        x_str[i] = 'X'
        pauli_x = ''.join(reversed(x_str))
        paulis.append(pauli_x)
        coeffs.append(hx)

        z_str = ['I'] * num_qubits
        z_str[i] = 'Z'
        pauli_z = ''.join(reversed(z_str))
        paulis.append(pauli_z)
        coeffs.append(hz)

    # Hamiltonian zusammensetzen
    hamiltonian = SparsePauliOp.from_list(list(zip(paulis, coeffs)))

    if return_paulis:
        return hamiltonian, paulis
    else:
        return hamiltonian
    
def edge_coloring(bonds, num_qubits):
    """
    Finde eine Kantenfärbung (edge coloring) der Bonds.
    Gibt ein dict: {layer: [(i,j), ...]} zurück.
    """
    layers = []
    for (i, j) in bonds:
        # kleinster Layer finden, wo weder i noch j benutzt wird
        placed = False
        for layer in layers:
            used_qubits = {q for edge in layer for q in edge}
            if i not in used_qubits and j not in used_qubits:
                layer.append((i, j))
                placed = True
                break
        if not placed:
            layers.append([(i, j)])
    return {ell: layer for ell, layer in enumerate(layers)}

def build_hva_layers(topology=None, num_layers=1):
    num_qubits, bonds = get_sites_and_bonds(topology)

    qc = QuantumCircuit(num_qubits)

    # Initialzustand: |+>^N durch H auf allen Qubits
    qc.h(range(num_qubits))

    # Parametervektor: für jeden Layer alpha, beta, gamma
    params = ParameterVector("θ", 3 * num_layers)

    bond_layers = edge_coloring(bonds, num_qubits)

    # Schleife über die Layer
    for layer in range(num_layers):
        alpha = params[3 * layer]
        beta  = params[3 * layer + 1]
        gamma = params[3 * layer + 2]

        # Z-Feld (longitudinal)
        for i in range(num_qubits):
            qc.rz(beta, i)

        # X-Feld (transversal)
        for i in range(num_qubits):
            qc.rx(gamma, i)

        # ZZ-Terme (nur für gekoppelte Qubits)
        for _, layer_bonds in bond_layers.items():
            for i, j in layer_bonds:
                qc.cx(i, j)
            for i, j in layer_bonds:
                qc.rz(alpha, j)
            for i, j in layer_bonds:
                qc.cx(i, j)

    return qc, params

def init_parameters(hx, hz, J, num_layers, c=0.3, jitter=0.02, seed=None):
    """
    Erzeugt initiale Parameter (alpha, beta, gamma) für num_layers.
    
    Args:
        hx, hz, J   : Hamilton-Parameter
        num_layers  : Anzahl der Ansatz-Layer
        c           : Gesamtskala für die Winkel (in rad)
        jitter      : Amplitude des Zufallsrauschens (in rad)
        seed        : Zufalls-Seed für Reproduzierbarkeit
    
    Returns:
        np.ndarray mit Shape (3*num_layers,), 
        Reihenfolge: [alpha_0, beta_0, gamma_0, alpha_1, ...]
    """
    rng = np.random.default_rng(seed)

    # Normalisierung an stärkstem Term
    s = max(abs(hx), abs(hz), abs(J), 1e-12)

    base_alpha = c * (J / s)
    base_beta  = c * (hz / s)
    base_gamma = c * (hx / s)

    params = []
    for l in range(num_layers):
        # gleichmäßig auf Layers verteilen
        alpha = base_alpha / num_layers
        beta  = base_beta / num_layers
        gamma = base_gamma / num_layers

        # kleines Rauschen hinzufügen
        alpha += rng.uniform(-jitter, jitter)
        beta  += rng.uniform(-jitter, jitter)
        gamma += rng.uniform(-jitter, jitter)

        params.extend([alpha, beta, gamma])

    return np.array(params)

################################################################################################################
# Compute energy
################################################################################################################

def optimize_energy(initial_params, Hamiltonian, topology, num_layers=1,
                    backend=None, mode="statevector", initial_layout=None,
                    maxiter=200, shots=1000, method="Powell", bounds=None):

    qc, param_labels = build_hva_layers(topology=topology, num_layers=num_layers)

    if mode == "measurement":
        paulis = [p.to_label() for p in Hamiltonian.paulis]
        coeffs = Hamiltonian.coeffs
        groups = group_paulis(paulis)
        bases  = determine_measurement_basis(groups)
        circuits = apply_measurement_bases(qc, bases)
        tcircs = transpile(circuits, backend, initial_layout=initial_layout)

        sampler = BackendSamplerV2(backend=backend, options={"default_shots": shots})

        def compute_energy(params):
            bind_map = {p: float(v) for p, v in zip(param_labels, params)}
            circiuits_bound = [qc.assign_parameters(bind_map) for qc in tcircs]
            res = sampler.run(circiuits_bound).result()
            counts_all = [r.data.meas.get_counts() for r in res]
            pauli_exp = get_pauli_expectation_dict(groups, counts_all)
            E = sum(pauli_exp[label] * c for label, c in zip(paulis, coeffs))
            return float(np.real(E))

    else:
        def compute_energy(params):
            bind_map = {p: float(v) for p, v in zip(param_labels, params)}
            psi = Statevector.from_instruction(qc.assign_parameters(bind_map))
            return float(np.real(psi.expectation_value(Hamiltonian)))

    kwargs = {"method": method, "options": {"maxiter": maxiter}}
    if bounds is not None:
        if method.upper() in {"L-BFGS-B","TNC","SLSQP","POWELL","TRUST-CONSTR"}:
            kwargs["bounds"] = bounds
        elif method.upper() == "COBYLA":
            cons = []
            for k, (lo, hi) in enumerate(bounds):
                cons += [{"type":"ineq","fun":(lambda x,k=k,lo=lo: x[k]-lo)},
                         {"type":"ineq","fun":(lambda x,k=k,hi=hi: hi-x[k])}]
            kwargs["constraints"] = cons
        else:
            raise ValueError(f"Bounds werden von '{method}' nicht unterstützt.")
        
    print("Optimierung startet...")
    result = minimize(compute_energy, initial_params, **kwargs)
    print("Optimierung fertig.")
    
    if mode == "measurement":
        bind_map = {p: float(v) for p, v in zip(param_labels, result.x)}
        circiuits_bound = [qc.assign_parameters(bind_map) for qc in tcircs]
        res = sampler.run(circiuits_bound).result()
        counts_all = [r.data.meas.get_counts() for r in res]
        pauli_exp = get_pauli_expectation_dict(groups, counts_all)
        
        HelperInfo = {
            "circuit": qc,
            "paulis": paulis,
            "groups":groups,
            "bases":bases,
            "exp_values": pauli_exp,
            "counts": counts_all,
            "initial_params": initial_params,
            "opt_params": result.x,
            "energy": result.fun
        }
        return result.x, result.fun, HelperInfo
    else:
        return result.x, result.fun, None


###############################################################################################################
# Run scan
###############################################################################################################

def run_scan(J=-1, num_layers=1, topology=None, N=20, maxiter=300, method="Powell", bounds=None, mode= "statevector", backend=None, initial_layout=None):
    """
    Scannt hx,hz auf einem N×N-Gitter und gibt ein Dictionary mit allen Resultaten zurück.

    Args:
        J, num_layers, topology : Hamilton-/Ansatz-Parameter
        N        : Anzahl Punkte pro Achse (hx,hz in [-2,2])
        mode     : "statevector" oder "measurement" (wird an optimize_energy durchgereicht)
        maxiter  : Max-Iterationen für den Optimierer

    Returns:
        results: dict mit Schlüsseln
            - "hx_vals", "hz_vals"
            - "grid": {"Hx", "Hz"}  # Meshgrids
            - "Energy_exact", "Energy_vqe", "Rel_error"
            - "Opt_parameters"  # shape (N, N, 3*num_layers)
            - "max_error", "max_error_index", "max_error_coords"
            - "settings"
    """
    # Gitterachsen
    hx_vals = np.linspace(-2, 2, N)
    hz_vals = np.linspace(-2, 2, N)

    # Speicher
    Energy_exact   = np.zeros((N, N))
    Energy_vqe     = np.zeros((N, N))
    Rel_error      = np.zeros((N, N))
    Opt_parameters = np.zeros((N, N, 3 * num_layers))

    total = len(hx_vals) * len(hz_vals)
    pbar = tqdm(total=total, desc="Scanning (hx,hz)")

    # Scan
    for i, hx in enumerate(hx_vals):
        for j, hz in enumerate(hz_vals):

            # Hamiltonian
            H = MFIM_Hamiltonian(J=J, hx=hx, hz=hz, topology=topology)

            # Exakte Grundenergie
            exact_energy = np.min(eigh(H.to_matrix(), eigvals_only=True))
            Energy_exact[i, j] = exact_energy

            # Startwerte
            init_params = init_parameters(hx, hz, J, num_layers)

            # VQE
            opt_params, energy = optimize_energy(
                initial_params=init_params,
                Hamiltonian=H,
                topology=topology,
                num_layers=num_layers,
                maxiter=maxiter,
                method=method, 
                bounds=bounds,
                mode = mode,
                backend= backend,
                initial_layout=initial_layout
            )

            # Speichern
            Energy_vqe[i, j]       = energy
            Opt_parameters[i, j, :] = opt_params
            denom = max(abs(exact_energy), 1e-12)  # robust gegen 0
            Rel_error[i, j]        = abs(energy - exact_energy) / denom
            pbar.update(1)

    # Maximalen Fehler finden
    flat_idx = np.nanargmax(Rel_error)
    i_max, j_max = np.unravel_index(flat_idx, Rel_error.shape)
    err_max = Rel_error[i_max, j_max]
    hx_max  = hx_vals[i_max]
    hz_max  = hz_vals[j_max]

    # Ergebnis-Dictionary
    Hx, Hz = np.meshgrid(hx_vals, hz_vals, indexing="ij")
    results = {
        "hx_vals": hx_vals,
        "hz_vals": hz_vals,
        "grid": {"Hx": Hx, "Hz": Hz},
        "Energy_exact": Energy_exact,
        "Energy_vqe": Energy_vqe,
        "Rel_error": Rel_error,
        "Opt_parameters": Opt_parameters,          # shape (N,N, 3*num_layers)
        "max_error": float(err_max),
        "max_error_index": (int(i_max), int(j_max)),
        "max_error_coords": {"hx": float(hx_max), "hz": float(hz_max)},
        "settings": {
            "J": J,
            "num_layers": num_layers,
            "N": N,
            "maxiter": maxiter,
            "topology": topology
        }
    }
    pbar.close()
    return results


###############################################################################################################
# Functions for efficient measurements
###############################################################################################################

def locally_commutable(p1, p2):
    """
    Prüft ob zwei Pauli-Strings lokal messbar sind: 
    auf keinem Qubit gibt es gleichzeitig z.B. X und Y.
    """
    for a, b in zip(p1, p2):
        if a == 'I' or b == 'I':
            continue
        if a != b:
            return False  # unterschiedliche Nicht-I-Terms → nicht lokal gleichzeitig messbar
    return True

def group_paulis(pauli_strings):
    """
    Gruppiert Pauli-Strings in Gruppen mit gemeinsamer Messbasis (lokal messbar).
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(pauli_strings)))

    # Kante falls sie NICHT gemeinsam messbar sind (Konfliktgraph)
    for i in range(len(pauli_strings)):
        for j in range(i + 1, len(pauli_strings)):
            if not locally_commutable(pauli_strings[i], pauli_strings[j]):
                G.add_edge(i, j)

    # Graph Coloring
    coloring = nx.coloring.greedy_color(G, strategy="largest_first")

    # Gruppieren nach Farben
    groups = {}
    for idx, color in coloring.items():
        groups.setdefault(color, []).append(pauli_strings[idx])

    return list(groups.values())


def determine_measurement_basis(pauli_groups):
    """
    Bestimmt eine Messbasis (als Liste von 'X', 'Y', 'Z', 'I') für eine Gruppe kommutierender Pauli-Strings.
    Die zurückgegebene Basis diagonalisiert alle Paulis in der Gruppe gleichzeitig.
    """
    
    bases = []
    for group in pauli_groups:
        num_qubits = len(group[0])
        basis = ['Z'] * num_qubits  # Default: Messung in Z

        for qubit in range(num_qubits):
            paulis_on_qubit = set(p[qubit] for p in group if p[qubit] != 'I')

            if not paulis_on_qubit:
                basis[qubit] = 'I'
            elif paulis_on_qubit == {'Z'}:
                basis[qubit] = 'Z'
            elif paulis_on_qubit == {'X'}:
                basis[qubit] = 'X'
            elif paulis_on_qubit == {'Y'}:
                basis[qubit] = 'Y'
            else:
                # Mehrere unterschiedliche Paulis (X,Y,Z) → nicht gemeinsam diagonal.
                # In Gruppenbildung sollte das nicht vorkommen.
                raise ValueError(f"Nicht kompatible Paulis auf Qubit {qubit}: {paulis_on_qubit}")
            
        bases.append(basis)

    return bases

def apply_measurement_bases(base_circuit, measurement_bases):
    """
    Baut pro Messbasis einen Circuit: Rotationen in Z-Basis + Messung.
    Erwartet: measurement_bases[i][q] ist Basis für Qubit q.
    """
    circuits = []
    for basis in measurement_bases:
        qc = base_circuit.copy()
        for q, b in enumerate(basis):
            if b == 'X':
                qc.h(q)
            elif b == 'Y':
                qc.sdg(q); qc.h(q)
            # 'Z' oder 'I': nix tun
        qc.measure_all()
        circuits.append(qc)
    return circuits


def get_expectation(pauli: str, counts: dict) -> float:
    """
    Erwartungswert für einen Pauli-String aus Counts.
    Achtung: Pauli-Label ist big-endian, counts-Bitstring: rechtes Bit = Qubit 0.
    """
    shots = sum(counts.values())
    if shots == 0:
        return 0.0

    n = len(pauli)
    exp = 0.0
    for bitstring, c in counts.items():
        parity = 0
        for q in range(n):
            if pauli[n-1-q] != 'I':                 # Label: Index n-1-q ↔ Qubit q
                bit = int(bitstring[-1 - q])        # Counts: rechtes Bit ↔ Qubit 0
                parity ^= bit
        exp += (1 if parity == 0 else -1) * c

    val = exp / shots
    return float(val)


def get_pauli_expectation_dict(groups, counts):
    pauli_expectations = {}

    for i in range(len(groups)):
        for pauli_str in groups[i]:
            exp_val = get_expectation(pauli=pauli_str, counts=counts[i])
            pauli_expectations[pauli_str] = exp_val

    return pauli_expectations


def compute_exp_value(qc,Hamiltonian,backend,phys_qubits=None, shots=1024):

    sampler = BackendSamplerV2(backend=backend, options={"default_shots": shots})
    cmap = CouplingMap(couplinglist=[
                (u, v)
                for (u, v) in backend.configuration().coupling_map
                if u in phys_qubits and v in phys_qubits
            ])

    paulis = [p.to_label() for p in Hamiltonian.paulis]
    coeffs = Hamiltonian.coeffs
    groups = group_paulis(paulis)
    bases  = determine_measurement_basis(groups)
    circuits = apply_measurement_bases(qc, bases)
    tcircs = transpile(circuits=circuits, initial_layout=phys_qubits, coupling_map=cmap)

    res = sampler.run(tcircs).result()
    counts_all = [r.data.meas.get_counts() for r in res]
    pauli_exp = get_pauli_expectation_dict(groups, counts_all)
    exp_val = sum(pauli_exp[label] * c for label, c in zip(paulis, coeffs))

    return exp_val