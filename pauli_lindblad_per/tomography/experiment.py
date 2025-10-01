from tomography.processorspec import ProcessorSpec
from tomography.layerlearning import LayerLearning
from tomography.analysis import Analysis
from framework.percircuit import PERCircuit
from per.perexperiment import PERExperiment

from typing import List, Any
import logging
import datetime
 
logging.basicConfig(filename="experiment.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger = logging.getLogger("experiment")
logger.setLevel(logging.INFO)

from primitives.circuit import QiskitCircuit
from primitives.processor import QiskitProcessor
import pickle
import qiskit

class SparsePauliTomographyExperiment:
    """This class carries out the full experiment by creating and running a LayerLearning
    instance for each distinct layer, running the analysis, and then returning a PERCircuit
    with NoiseModels attached to each distinct layer"""

    def __init__(self, circuits, inst_map, backend, used_qubits):

        circuit_interface = None
        subgraph = False
        if len(used_qubits) != backend.num_qubits:
            subgraph = True
            
        self.subgraph= subgraph
        
        if circuits[0].__class__.__name__ == "QuantumCircuit":
            circuit_interface = QiskitCircuit
            processor = QiskitProcessor(backend, subgraph=subgraph)
        else:
            raise Exception("Unsupported circuit type")
    
        self._profiles = set()
        for circuit in circuits: 
            circ_wrap = circuit_interface(circuit)
            parsed_circ = PERCircuit(circ_wrap)
            for layer in parsed_circ._layers:
                if layer.cliff_layer:
                    self._profiles.add(layer.cliff_layer)

        logger.info("Generated layer profile with %s layers:"%len(self._profiles))
        for layer in self._profiles:
            logger.info(layer)

        self._procspec = ProcessorSpec(inst_map, processor, used_qubits)
        self.instances = []
        self._inst_map = inst_map
        self._layers = None
        self.used_qubits = used_qubits

        self._layers = []
        for l in self._profiles:
            learning = LayerLearning(l,self._procspec)
            self._layers.append(learning)

        self.analysis = Analysis(self._layers, self._procspec)

    def generate(self, samples, single_samples, depths):
        """This method is used to generate the experimental benchmarking procedure. The samples
        are the number of times to sample from the Pauli twirl. The single_samples controls
        how many twirl samples to take from the degeneracy-lifting measurements. It may desirable
        to make this higher since the error on these measurements will generally be higher.
        The depths control the different circuit depths to use for the exponential fits."""

        if len(depths) < 2:
            raise Exception("Exponental fit requires 3 or more depth data points.")

        for l in self._layers:
            l.procedure(samples, single_samples, depths)

    def run(self, executor, auto_save=True, save_filename=None):
        """This method produces a list of circuits in the native representation, passes them 
        as a list to the executor method, and associates the result with the benchmark instances
        that produced it
        
        Args:
            executor: Function that executes circuits and returns results
            auto_save: If True, automatically save results after execution
            save_filename: Custom filename for saving (default: experiment_results_{timestamp}.pkl)
        """

        instances = []
        for l in self._layers:
            instances += l.instances

        circuits = [inst.get_circuit() for inst in instances]
        results = executor(circuits)

        for res,inst in zip(results, instances): #TODO: find out if order can be preserved
            inst.add_result(res)
        
        # Auto-save results after successful execution
        if auto_save:
            if save_filename is None:
                import os
                # Ensure SaveFiles directory exists
                os.makedirs("SaveFiles", exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_filename = f"SaveFiles/experiment_results_{timestamp}.json"
            
            try:
                self.save(save_filename)
                logger.info(f"Results automatically saved to {save_filename}")
            except Exception as e:
                logger.error(f"Failed to auto-save results: {e}")
                logger.error("Continuing without saving - results are still in memory")

    def analyze(self):
        """Runs analysis on each layer representative and stores for later plotting/viewing"""
        self.analysis.analyze()
        return self.analysis.noisedataframe

    def create_per_experiment(self, circuits : Any) -> PERExperiment:
        experiment = PERExperiment(circuits, self._inst_map, self.analysis.noisedataframe, backend = None, procspec = self._procspec)
        return experiment

    def save(self, filename=None):
        """Save experiment results and state to file for crash recovery
        
        Args:
            filename: Filename to save to (supports .pkl, .json)
        """
        import pickle
        import json
        import os
        
        # Set default filename in SaveFiles directory
        if filename is None:
            os.makedirs("SaveFiles", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"SaveFiles/experiment_results_{timestamp}.json"
        
        # Prepare data for saving
        save_data = {
            'timestamp': str(datetime.datetime.now()),
            'experiment_config': {
                'inst_map': self._inst_map,
                'used_qubits': self.used_qubits,
                'subgraph': self.subgraph,
                'num_profiles': len(self._profiles)
            },
            'layer_results': [],
            'metadata': {
                'framework_version': '1.0',  # Could be extracted from package
                'qiskit_version': qiskit.__version__ if 'qiskit' in globals() else 'unknown'
            }
        }
        
        # Save results from each layer
        for layer_idx, layer in enumerate(self._layers):
            layer_data = {
                'layer_index': layer_idx,
                'instances': []
            }
            
            # Save instance results 
            for inst_idx, inst in enumerate(layer.instances):
                if hasattr(inst, '_result') and inst._result is not None:
                    instance_data = {
                        'instance_index': inst_idx,
                        'result': inst._result,
                        'metadata': getattr(inst, 'metadata', {}),
                        # Save essential instance info for reconstruction
                        'meas_basis': str(inst._meas_basis) if hasattr(inst, '_meas_basis') else None,
                        'depth': getattr(inst, 'depth', None),
                        'type': getattr(inst, 'type', None)
                    }
                    layer_data['instances'].append(instance_data)
            
            save_data['layer_results'].append(layer_data)
        
        # Save to file
        file_extension = os.path.splitext(filename)[1].lower()
        
        try:
            if file_extension == '.pkl':
                with open(filename, 'wb') as f:
                    pickle.dump(save_data, f)
            else:  # Default to JSON
                if not filename.endswith('.json'):
                    filename += '.json'
                with open(filename, 'w') as f:
                    json.dump(save_data, f, indent=2, default=str)
                    
            logger.info(f"Experiment results saved to {filename}")
            logger.info(f"Saved {sum(len(layer['instances']) for layer in save_data['layer_results'])} instance results")
            
        except Exception as e:
            logger.error(f"Failed to save experiment results: {e}")
            raise

    def load(self, filename=None, restore_results=True):
        """Load experiment results from file for crash recovery
        
        Args:
            filename: Filename to load from
            restore_results: If True, restore results to instance objects
            
        Returns:
            dict: Loaded data dictionary
        """
        import pickle
        import json
        import os
        
        # Set default filename in SaveFiles directory
        if filename is None:
            # Try to find the most recent experiment file
            import glob
            # Look for both JSON and PKL files, prefer JSON
            json_pattern = "SaveFiles/experiment_results_*.json"
            pkl_pattern = "SaveFiles/experiment_results_*.pkl"
            json_files = glob.glob(json_pattern)
            pkl_files = glob.glob(pkl_pattern)
            
            all_files = json_files + pkl_files
            if all_files:
                filename = max(all_files, key=os.path.getmtime)  # Most recent file
                logger.info(f"No filename specified, using most recent: {filename}")
            else:
                raise FileNotFoundError("No experiment files found in SaveFiles/ directory")
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Save file not found: {filename}")
        
        file_extension = os.path.splitext(filename)[1].lower()
        
        try:
            if file_extension == '.pkl':
                with open(filename, 'rb') as f:
                    save_data = pickle.load(f)
            else:  # Default to JSON
                with open(filename, 'r') as f:
                    save_data = json.load(f)
            
            logger.info(f"Loaded experiment results from {filename}")
            logger.info(f"Timestamp: {save_data.get('timestamp', 'unknown')}")
            
            # Restore results to instances if requested
            if restore_results:
                self._restore_results_from_data(save_data)
                
            return save_data
            
        except Exception as e:
            logger.error(f"Failed to load experiment results: {e}")
            raise

    def _restore_results_from_data(self, save_data):
        """Internal method to restore results to instance objects"""
        restored_count = 0
        
        for layer_data in save_data['layer_results']:
            layer_idx = layer_data['layer_index']
            
            if layer_idx < len(self._layers):
                layer = self._layers[layer_idx]
                
                for instance_data in layer_data['instances']:
                    inst_idx = instance_data['instance_index']
                    
                    if inst_idx < len(layer.instances):
                        inst = layer.instances[inst_idx]
                        
                        # Restore the result
                        if 'result' in instance_data:
                            inst.add_result(instance_data['result'])
                            restored_count += 1
                        
                        # Restore metadata if available
                        if 'metadata' in instance_data:
                            for key, value in instance_data['metadata'].items():
                                setattr(inst, key, value)
        
        logger.info(f"Restored results for {restored_count} instances")
        
    def analyze_from_file(self, filename=None):
        """Convenience method to load results and run analysis in one step
        
        Args:
            filename: Filename to load results from
            
        Returns:
            Analysis results (same as analyze())
        """
        logger.info(f"Loading results from {filename} and running analysis...")
        self.load(filename, restore_results=True)
        return self.analyze()

    def has_results(self):
        """Check if experiment has results loaded"""
        for layer in self._layers:
            for inst in layer.instances:
                if hasattr(inst, '_result') and inst._result is not None:
                    return True
        return False
