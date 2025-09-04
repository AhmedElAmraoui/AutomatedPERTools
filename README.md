# AutomatedPERTools

Dieses Repository enthält eine Sammlung von Modulen und Experimentier-Tools für
**Tomography** und **Probabilistic Error Reduction (PER)** Experimente auf
Quanten-Hardware und Simulatoren.  

Ziel ist es, automatisiert Circuits zu generieren, sie mit Twirling-Strategien
auszuführen, und die Ergebnisse zu analysieren. Zusätzlich gibt es Hilfsfunktionen,
um die Anzahl der tatsächlich erzeugten Circuits (und optional Backend-Runs inkl.
Shots) im Voraus zu berechnen.

---

## Projektstruktur

- **`primitives/`**  
  Low-Level-Bausteine und Schnittstellen zu Qiskit:
  - `processor.py` – Definition von `QiskitProcessor` (Backend-Anbindung)
  - `circuit.py` – Wrapper für `QuantumCircuit` → interne Circuit-API

- **`framework/`**  
  Framework-Komponenten für die Repräsentation von Circuits:
  - `percircuit.py` – Zerlegung von Circuits in einzelne Layer (`cliff_layer`)

- **`tomography/`**  
  Module für Sparse Pauli Tomography:
  - `experiment.py` – `SparsePauliTomographyExperiment`, steuert Generierung, Ausführung, Analyse
  - `layerlearning.py` – Berechnung der kompatiblen Messbasen je Layer (`_single_bases`)
  - `processorspec.py` – Spezifikation des Prozessors (Gate-Map, Messbasen)
  - `layernoisedata.py` – Noise-Modellierung auf Layer-Ebene

- **`per/`**  
  Module für Probabilistic Error Reduction:
  - `perexperiment.py` – PER-Experiment-Klasse, Generierung & Analyse
  - `perrun.py`, `perinstance.py` – Helfer für einzelne Runs und Instanzen

- **`src/`**  
  Zusätzliche Hilfsfunktionen:
  - `calculate_runs.py` – Funktionen zur Berechnung der Anzahl erzeugter Circuits und Backend-Runs
    für Tomography- und PER-Experimente.

---

## Installation

Voraussetzung: Python 3.10+, [Qiskit](https://qiskit.org), sowie ggf. weitere Pakete
wie `numpy`, `scipy`.

```bash
git clone https://github.com/<user>/AutomatedPERTools.git
cd AutomatedPERTools
pip install -r requirements.txt