from scipy.optimize import nnls
import numpy as np
from matplotlib import pyplot as plt

from framework.noisemodel import NoiseModel
from tomography.layerlearning import LayerLearning
from tomography.termdata import TermData, COLORS
from primitives.circuit import Circuit
from tomography.benchmarkinstance import BenchmarkInstance, SINGLE, PAIR

import logging
from itertools import cycle
from qiskit.quantum_info import Pauli

logger = logging.getLogger("experiment")

class LayerNoiseData:
    """This class is responsible for aggregating the data associated with a single layer,
    processing it, and converting it into a noise model to use for PER"""

    def __init__(self, layer : LayerLearning):
        self._term_data = {} #keys are terms and the values are TermDatas
        self.layer = layer

        for pauli in layer._procspec.model_terms:
            pair = layer.pairs[pauli]
            self._term_data[pauli] = TermData(pauli, pair)

    def sim_meas(self, pauli):
        """Given an instance and a pauli operator, determine how many terms can be measured"""
        return [term for term in self.layer._procspec.model_terms if pauli.simultaneous(term)]

    def single_sim_meas(self, pauli, prep):
        return [term for pair,term in self.layer.single_pairs if pauli.simultaneous(term) and prep.simultaneous(pair)]

    def add_expectations(self):
        for inst in self.layer.instances:
            self.add_expectation(inst)

    def add_expectation(self, inst : BenchmarkInstance):
        """Add the result of a benchmark instance to the correct TermData object"""

        basis = inst.meas_basis
        prep = inst.prep_basis
        pair_sim_meas = {}
        single_sim_meas = {}

        if inst.type == SINGLE: 

            if not basis in single_sim_meas:
                single_sim_meas[basis] = self.single_sim_meas(basis, prep)

            for pauli in single_sim_meas[basis]:
                expectation = inst.get_expectation(pauli)
                self._term_data[pauli].add_single_expectation(expectation)

        elif inst.type == PAIR:

            if not basis in pair_sim_meas:
                pair_sim_meas[basis] = self.sim_meas(basis)

            for pauli in pair_sim_meas[basis]:
                #add the expectation value to the data for this term
                expectation = inst.get_expectation(pauli)
                self._term_data[pauli].add_expectation(inst.depth, expectation, inst.type)

        
    def fit_noise_model(self):
        """Fit all of the terms, and then use obtained SPAM coefficients to make degerneracy
        lifting estimates"""

        for term in self._term_data.values(): #perform all pairwise fits
            term.fit()
        
        for pair,pauli in self.layer.single_pairs:
            self._term_data[pauli].fit_single()
            pair_dat = self._term_data[pair]
            pair_dat.fidelity = pair_dat.fidelity**2/self._term_data[pauli].fidelity

        
        logger.info("Fit noise model with following fidelities:") 
        logger.info([term.fidelity for term in self._term_data.values()])

        #get noise model from fits
        self.nnls_fit()

    def _issingle(self, term):
        return term.pauli != term.pair and term.pair in self._term_data
  
    
    def nnls_fit(self):
        """Generate a noise model corresponding to the Clifford layer being benchmarked
        for use in PER"""

        def sprod(a,b): #simplecting inner product between two Pauli operators
            return int(not a.commutes(b))

        F1 = [] #First list of terms
        F2 = [] #List of term pairs
        fidelities = [] # list of fidelities from fits

        for datum in self._term_data.values():
            F1.append(datum.pauli)
            fidelities.append(datum.fidelity)
            #If the Pauli is conjugate to another term in the model, a degeneracy is present
            if self._issingle(datum):
                F2.append(datum.pauli)
            else:
                pair = datum.pair
                F2.append(pair)

        #create commutativity matrices
        M1 = [[sprod(a,b) for a in F1] for b in F1]
        M2 = [[sprod(a,b) for a in F1] for b in F2]

        #check to make sure that there is no degeneracy
        if np.linalg.matrix_rank(np.add(M1,M2)) != len(F1):
            raise Exception("Matrix is not full rank, something went wrong!")
       
        #perform least-squares estimate of model coefficients and return as noisemodel 
        coeffs,_ = nnls(np.add(M1,M2), -np.log(fidelities)) 
        self.noisemodel = NoiseModel(self.layer._cliff_layer, F1, coeffs)

    def _model_terms(self, links): #return a list of Pauli terms with the specified support
        groups = []
        for link in links:
            paulis = []
            for pauli in self._term_data.keys():
                overlap = [pauli[q].to_label() != "I" for q in link]
                support = [p.to_label() == "I" or q in link for q,p in enumerate(pauli)]
                if all(overlap) and all(support):
                    paulis.append(pauli)
            groups.append(paulis)

        return groups

    def get_spam_coeffs(self):
        """Return a dictionary of the spam coefficients of different model terms for use in 
        readout error mitigation when PER is carried out."""

        return dict(zip(self._term_data.keys(), [termdata.spam for termdata in self._term_data.values()]))

    def plot_coeffs(self, *links, plot_style = 1):
        """Plot the model coefficients in the generator of the sparse model corresponding
        to the current circuit layer"""
        
        if plot_style == 1:
            coeffs_dict = dict(self.noisemodel.coeffs)
            groups = self._model_terms(links)
            fig, ax = plt.subplots()
            colcy = cycle(COLORS)
            for group in groups:
                c = next(colcy)
                coeffs = [coeffs_dict[term] for term in group]
                ax.bar([term.to_label() for term in group], coeffs, color=c)
        elif plot_style == 2:
            model_terms_x = list(self.layer._procspec.model_terms)
            model_terms = []
            for term in model_terms_x:
                model_terms.append(str(term))
            coeffs_dict = dict(self.noisemodel.coeffs)
            coeffs = {
            term.to_label(): value
            for term, value in self.noisemodel.coeffs}
            coupling_list = self.layer._procspec._processor.sub_map(self.layer._procspec.inst_map)
            self.plot_grouped_by_qubit(model_terms, coeffs, coupling_list, title="Fidelity", ylabel= "Coefficients")

    def graph(self, *links):
        """Graph the fits values for a certain subset of Pauli terms"""

        groups = self._model_terms(links)
        fig, ax = plt.subplots()
        for group in groups:
            for term in group:
                termdata = self._term_data[term]
                termdata.graph(ax)

        return ax

    def plot_infidelitites(self, *links, plot_style = 1):
        """Plot the infidelities of a subset of Pauli terms"""
        if plot_style == 1:
            groups = self._model_terms(links)
            fig, ax = plt.subplots()
            colcy = cycle(COLORS)
            for group in groups:
                c = next(colcy)
                infidelities = [1-self._term_data[term].fidelity for term in group]
                ax.bar([term.to_label() for term in group], infidelities, color=c)
            return ax
        elif plot_style == 2:
            model_terms_x = list(self.layer._procspec.model_terms)
            model_terms = []
            for term in model_terms_x:
                model_terms.append(str(term))
            infidelities = {}
            for term in model_terms:
                infidelities[term] = 1-self._term_data[Pauli(term)].fidelity
            
            coupling_list = self.layer._procspec._processor.sub_map(self.layer._procspec.inst_map)
            self.plot_grouped_by_qubit(model_terms, infidelities, coupling_list)
            
            
    def plot_grouped_by_qubit(self, model_terms, coeffs, coupling_list, title="Fidelity", ylabel= "Coefficients"):
        """
        Plots measured vs ideal fidelities grouped by qubit or qubit pair,
        sorted alphabetically within groups, with clean Pauli labels and separators.
        """
        n_qubits = len(str(model_terms[1]))
        allowed_pairs = {(min(a, b), max(a, b)) for (a, b) in coupling_list.edge_list()}

        block_colors = ['#d0e1f9', '#f9d0d0', '#d0f9d9', '#f9f5d0', '#e0d0f9', '#f0c0f9']

        def pauli_support(pauli):
            return [i for i, p in enumerate(pauli) if p != 'I']

        def compact_pauli_label(pauli):
            return ''.join(p for p in pauli if p != 'I')

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
                bar_measured.append(coeffs[model_terms[idx]])
                current_index += 1

            separator_lines.append(current_index - 0.5)

        # Balken plotten
        bar_positions = np.array(bar_positions)
        ax.bar(bar_positions, bar_measured, 0.4, color='tab:blue', label='Measured')
        
        from matplotlib.patches import Rectangle
        
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