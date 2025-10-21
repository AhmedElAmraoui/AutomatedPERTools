from pauli_lindblad_per.primitives.circuit import QiskitCircuit
from pauli_lindblad_per.framework.percircuit import PERCircuit
from pauli_lindblad_per.per.perrun import PERRun
from pauli_lindblad_per.primitives.processor import QiskitProcessor
import datetime
import logging

class PERExperiment:
    """This class plays the role of the SparsePauliTomographyExperiment class but for the
    generation, aggregation, and analysis of PER database
    
    class functions:
    - get the minimal number of measurement bases required to reconstruct desired observables
    - initialize generation of PER circuits to estimate each expectation for each circuit
        at the desired noise strength 
    - Pass the circuits to user-defined run method for execution of
    - Process results and return for display
    """
    
    def __init__(self, circuits, noise_data_frame, backend = None, procspec = None):
        """Initializes a PERExperiment with the data that stays constant for all circuits/
        noise strengths/expectation values

        Args:
            circuits (Any): Circuits to run with PER
            noise_data_frame (NoiseDataFrame): Noise models learned from tomography
            backend (Any): Backend to use for transpilation. None if passing an initialize processor
            processor (Processor) : Backend to use for transpilation. None if passing a native backend
        """
        circuit_interface = None

        #check if circuits have implementable type, and initialize processor
        if circuits[0].__class__.__name__ == "QuantumCircuit":
            circuit_interface = QiskitCircuit
            if backend:
                self._processor = QiskitProcessor(backend)
        else:
            raise Exception("Unsupported circuit type")
        if not backend:
            self._processor = procspec._processor  
        self.pauli_type = circuit_interface(circuits[0]).pauli_type
        self.procspec = procspec


        self.noise_data_frame = noise_data_frame #store noise data
        #Geerate list of PER circuits and assign noise models to layers 
        per_circuits = []
        for circ in circuits:
            circ_wrap = circuit_interface(circ) #wrap Circuit object
            per_circ = PERCircuit(circ_wrap) #create a PER circuit
            per_circ.add_noise_models(noise_data_frame) #add associated noise models
            per_circuits.append(per_circ)

        self._per_circuits = per_circuits

    def get_meas_bases(self, expectations):
        """Return the minimal set of bases needed to reconstruct the desired expectation values

        Args:
            expectations (Pauli): The desired Pauli expectation values
        """

        meas_bases = []
        #iterate through expectations
        for pauli in expectations:
            for i,base in enumerate(meas_bases): #iterate through bases
                if base.nonoverlapping(pauli): #if nonoverlapping, compose into last basis
                    meas_bases[i] = base.get_composite(pauli)
                    break
            else:
                meas_bases.append(pauli) #if no break is reached, append to end

        self.meas_bases = meas_bases
        
    def generate(
        self, 
        expectations, 
        samples, 
        noise_strengths
        ):
        """Initiate the generation of circuits required for PER

        Args:
            noise_strengths (list[int]): strengths of noise for PER fit
            expectations (list[str]): expectation values to reconstruct
            samples (int): number of samples to take from distribution
        """

        #Convert string labels to Pauli representation
        expectations = [self.pauli_type(label) for label in expectations]

        #get minimal set of measurement bases
        self.get_meas_bases(expectations)
        bases = self.meas_bases

        self._per_runs = []
        #initialize PERRun for each PERCircuit
        for pcirc in self._per_circuits:
            per_run = PERRun(
                self._processor, 
                pcirc, 
                samples,
                noise_strengths,
                bases, 
                expectations,
                self.procspec
                )
            self._per_runs.append(per_run)

    def run(self, executor, auto_save=True, save_filename=None):
        """pass a list of circuit in the native language to the executor method and await results

        Args:
            executor (method): list of circuits -> Counter of results
            auto_save: If True, automatically save results after execution
            save_filename: Custom filename for saving (default: per_results_{timestamp}.pkl)
        """

        #aggregate all instances into a list
        instances = []
        for run in self._per_runs:
            instances += run.instances
       
        #get circuits in native representation
        circuits = [inst.get_circuit() for inst in instances] 

        #pass circuit to executor
        results = executor(circuits)
       
        #add results to instances 
        for inst, res in zip(instances, results):
            inst.add_result(res)
            
        # Auto-save results after successful execution
        if auto_save:
            if save_filename is None:
                import os
                # Ensure SaveFiles directory exists
                os.makedirs("SaveFiles", exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_filename = f"SaveFiles/per_results_{timestamp}.json"
            
            try:
                self.save(save_filename)
                logging.info(f"PER results automatically saved to {save_filename}")
            except Exception as e:
                logging.error(f"Failed to auto-save PER results: {e}")
                logging.error("Continuing without saving - results are still in memory")

    def analyze(self):

        #run analysis on all instances and return results for each circuit
        for run in self._per_runs:
            run.analyze()

        return self._per_runs

    def get_overhead(self, layer, noise_strength):
        return self._per_circuits[layer].overhead(noise_strength)
    
    def save(self, filename=None):
        """Save PER experiment results and state to file for crash recovery
        
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
            filename = f"SaveFiles/per_results_{timestamp}.json"
        
        # Prepare data for saving
        save_data = {
            'timestamp': str(datetime.datetime.now()),
            'experiment_type': 'PER',
            'experiment_config': {
                'num_per_runs': len(self._per_runs),
                'meas_bases': [str(base) for base in getattr(self, 'meas_bases', [])],
            },
            'per_run_results': [],
            'metadata': {
                'framework_version': '1.0',
            }
        }
        
        # Save results from each PER run
        for run_idx, per_run in enumerate(self._per_runs):
            run_data = {
                'run_index': run_idx,
                'instances': []
            }
            
            # Save instance results 
            for inst_idx, inst in enumerate(per_run.instances):
                if hasattr(inst, '_result') and inst._result is not None:
                    instance_data = {
                        'instance_index': inst_idx,
                        'result': inst._result,
                        'noise_strength': getattr(inst, 'noise_strength', None),
                        'meas_basis': str(getattr(inst, '_meas_basis', 'unknown')),
                        'metadata': getattr(inst, 'metadata', {})
                    }
                    run_data['instances'].append(instance_data)
            
            save_data['per_run_results'].append(run_data)
        
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
                    
            logging.info(f"PER experiment results saved to {filename}")
            logging.info(f"Saved {sum(len(run['instances']) for run in save_data['per_run_results'])} PER instance results")
            
        except Exception as e:
            logging.error(f"Failed to save PER experiment results: {e}")
            raise

    def load(self, filename=None, restore_results=True):
        """Load PER experiment results from file for crash recovery
        
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
            # Try to find the most recent PER file
            import glob
            # Look for both JSON and PKL files, prefer JSON
            json_pattern = "SaveFiles/per_results_*.json"
            pkl_pattern = "SaveFiles/per_results_*.pkl"
            json_files = glob.glob(json_pattern)
            pkl_files = glob.glob(pkl_pattern)
            
            all_files = json_files + pkl_files
            if all_files:
                filename = max(all_files, key=os.path.getmtime)  # Most recent file
                logging.info(f"No filename specified, using most recent: {filename}")
            else:
                raise FileNotFoundError("No PER files found in SaveFiles/ directory")
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"PER save file not found: {filename}")
        
        file_extension = os.path.splitext(filename)[1].lower()
        
        try:
            if file_extension == '.pkl':
                with open(filename, 'rb') as f:
                    save_data = pickle.load(f)
            else:  # Default to JSON
                with open(filename, 'r') as f:
                    save_data = json.load(f)
            
            logging.info(f"Loaded PER experiment results from {filename}")
            logging.info(f"Timestamp: {save_data.get('timestamp', 'unknown')}")
            
            # Restore results to instances if requested
            if restore_results:
                self._restore_per_results_from_data(save_data)
                
            return save_data
            
        except Exception as e:
            logging.error(f"Failed to load PER experiment results: {e}")
            raise

    def _restore_per_results_from_data(self, save_data):
        """Internal method to restore PER results to instance objects"""
        restored_count = 0
        
        for run_data in save_data['per_run_results']:
            run_idx = run_data['run_index']
            
            if run_idx < len(self._per_runs):
                per_run = self._per_runs[run_idx]
                
                for instance_data in run_data['instances']:
                    inst_idx = instance_data['instance_index']
                    
                    if inst_idx < len(per_run.instances):
                        inst = per_run.instances[inst_idx]
                        
                        # Restore the result
                        if 'result' in instance_data:
                            inst.add_result(instance_data['result'])
                            restored_count += 1
                        
                        # Restore metadata if available
                        if 'metadata' in instance_data:
                            for key, value in instance_data['metadata'].items():
                                setattr(inst, key, value)
                        
                        # Restore specific PER attributes
                        if 'noise_strength' in instance_data:
                            setattr(inst, 'noise_strength', instance_data['noise_strength'])
        
        logging.info(f"Restored results for {restored_count} PER instances")
        
    def analyze_from_file(self, filename=None):
        """Convenience method to load PER results and run analysis in one step
        
        Args:
            filename: Filename to load results from
            
        Returns:
            Analysis results (same as analyze())
        """
        logging.info(f"Loading PER results from {filename} and running analysis...")
        self.load(filename, restore_results=True)
        return self.analyze()

    def has_results(self):
        """Check if PER experiment has results loaded"""
        for per_run in self._per_runs:
            for inst in per_run.instances:
                if hasattr(inst, '_result') and inst._result is not None:
                    return True
        return False