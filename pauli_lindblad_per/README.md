# Pauli-Lindblad PER Framework Dokumentation

Dieses Framework implementiert ein Pauli Error Reconstruction (PER) System für Quantenschaltkreise basierend auf dem Pauli-Lindblad Formalismus. Das System besteht aus vier Hauptmodulen:

## Modulübersicht

### 1. Framework (`framework/`)
Das Framework-Modul stellt die Grundarchitektur für die Schaltkreisanalyse und Noise-Modellierung bereit.

### 2. PER (`per/`)
Das PER-Modul implementiert die eigentlichen Fehlerrekonstruktionsexperimente und Datenanalyse.

### 3. Primitives (`primitives/`)
Das Primitives-Modul definiert grundlegende Datenstrukturen und Abstraktionen für Quantenschaltkreise.

### 4. Tomografie (`tomography/`)
Das Tomografie-Modul führt die Charakterisierung von Quantengeräten durch und lernt Noise-Modelle.

---

## Framework Modul

### `circuitlayer.py`
Implementiert die `CircuitLayer` Klasse für die Zerlegung von Quantenschaltkreisen in Ebenen.

**Hauptfunktionen:**
- Trennung von Ein-Qubit-Gattern und Zwei-Qubit-Clifford-Gattern
- Pauli-Twirling für Noise-Minderung
- Sampling aus skalierten Noise-Repräsentationen

**Verwendung:**
```python
layer = CircuitLayer(circuit)
single_gates = layer.single_layer
clifford_gates = layer.cliff_layer
```

### `instance.py`
Basis-Framework für Benchmark- und PER-Instanzen mit gemeinsamer Funktionalität.

**Hauptfunktionen:**
- Basiswechsel für Messungen
- Readout-Twirling zur Reduzierung systematischer Fehler
- Erwartungswertberechnung mit Untwirling-Korrektur

### `noisemodel.py`
Verwaltung von Noise-Parametern und Implementierung von Noise-Skalierung.

**Hauptfunktionen:**
- Speicherung von Noise-Koeffizienten
- Probabilistische Implementierung der Quantum Probabilistic Decomposition (QPD)
- Berechnung des Sampling-Overheads

### `percircuit.py`
Aggregation von Schaltkreisebenen für PER-Experimente.

**Hauptfunktionen:**
- Automatische Zerlegung von Schaltkreisen in kompatible Ebenen
- Zuweisung von Noise-Modellen zu entsprechenden Ebenen
- Overhead-Berechnung für verschiedene Noise-Niveaus

---

## PER Modul

### `perdata.py`
Datensammlung und -analyse für PER-Experimente.

**Hauptfunktionen:**
- Aggregation von Erwartungswerten über verschiedene Noise-Niveaus
- Exponentieller Fit zur Extrapolation auf Noise-arme Grenzwerte
- Visualisierung der Noise-Skalierungscharakteristik

### `perexperiment.py`
Hauptkoordinator für PER-Experimente.

**Hauptfunktionen:**
- Automatische Generierung minimaler Messbasissets
- Koordination zwischen Schaltkreisgenerierung und -ausführung
- Integration mit Backend-Verarbeitungssystemen

### `perinstance.py` & `perrun.py`
Implementierung einzelner PER-Instanzen und Experimentdurchläufe.

---

## Primitives Modul

### `circuit.py`
Abstrakte Basisklasse für plattformunabhängige Quantenschaltkreise.

**Abstrakte Methoden:**
- `copy_empty()` - Leere Kopie des Schaltkreises erstellen
- `add_instruction()` - Quantenoperation hinzufügen
- `add_pauli()` - Pauli-Operator anhängen
- `compose()` - Schaltkreiskomposition
- `conjugate()` - Clifford-Konjugation für Pauli-Operatoren

**Qiskit-Implementierung:**
Die `QiskitCircuit` Klasse implementiert alle abstrakten Methoden für die Qiskit-Plattform.

### `instruction.py`, `pauli.py`, `processor.py`
Grundlegende Datenstrukturen für Quantenoperationen, Pauli-Operatoren und Hardware-Abstraktion.

---

## Tomografie Modul

### `experiment.py`
Hauptklasse für Sparse Pauli Tomografie Experimente.

**Workflow:**
1. Analyse der Eingabeschaltkreise zur Identifizierung einzigartiger Ebenen
2. Generierung von Benchmarking-Prozeduren für jede Ebene
3. Ausführung auf Quantenhardware
4. Analyse und Extraktion von Noise-Modellen
5. Erstellung von PER-Experimenten mit gelernten Modellen

### `analysis.py`
Datenanalyse für Tomografie-Experimente mit statistischer Auswertung.

### `layerlearning.py`
Maschinelles Lernen von Noise-Modellen für individuelle Schaltkreisebenen.

### `benchmarkinstance.py`
Implementierung von Benchmark-Instanzen für die Charakterisierung.

### Weitere Module:
- `layernoisedata.py` - Datenverwaltung für Ebenen-Noise
- `noisedataframe.py` - Strukturierte Speicherung von Noise-Daten
- `processorspec.py` - Hardware-Spezifikationen
- `termdata.py` - Verwaltung von Pauli-Term-Daten

---

## Anwendungsbeispiel

```python
# 1. Tomografie-Experiment initialisieren
experiment = SparsePauliTomographyExperiment(circuits, backend, used_qubits)

# 2. Benchmarking-Prozeduren generieren
experiment.generate(samples=1000, single_samples=2000, depths=[1, 2, 3, 4])

# 3. Auf Hardware ausführen
experiment.run(executor_function)

# 4. Noise-Modelle analysieren und lernen
noise_data_frame = experiment.analyze()

# 5. PER-Experiment erstellen
per_experiment = experiment.create_per_experiment(test_circuits)

# 6. PER mit gelernten Modellen durchführen
per_experiment.generate(expectations=['ZZII', 'XIXI'], samples=5000, noise_strengths=[0.1, 0.2, 0.3])
per_experiment.run(executor_function)
results = per_experiment.analyze()
```

## Systemarchitektur

Das System folgt einer modularen Architektur:

```
Eingabeschaltkreise
       ↓
Framework (Zerlegung in Ebenen)
       ↓
Tomografie (Noise-Charakterisierung) 
       ↓
PER (Fehlerminderung)
       ↓
Bereinigte Erwartungswerte
```
