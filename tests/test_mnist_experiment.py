import os
import unittest
from datetime import datetime
from omegaconf import OmegaConf

from experiment_manager.environment import Environment
from experiment_manager.experiment import Experiment
from experiment_manager.common.factory import Factory
from examples.pipelines.pipeline_example import TrainingPipeline


class TestMNISTExperiment(unittest.TestCase):
    def setUp(self):
        # Create a temporary workspace for testing
        self.workspace = os.path.abspath("test_workspace")
        os.makedirs(self.workspace, exist_ok=True)
        
        # Load experiment config
        self.config = OmegaConf.create({
            "name": "test_mnist",
            "id": 1,
            "desc": "Test MNIST experiment",
            "config_dir_path": os.path.join(os.path.dirname(__file__), "..", "examples", "mnist_experiment", "configs")
        })
        
        # Create factory
        self.factory = Factory()
        
        # Create environment
        self.env = Environment(
            workspace=self.workspace,
            config=OmegaConf.create({
                "verbose": True,
                "device": "cpu"
            }),
            factory=self.factory
        )
        self.env.setup_environment()
        
        # Create experiment
        self.experiment = Experiment.from_config(self.config, self.env)
        
    def tearDown(self):
        # Clean up test workspace
        import shutil
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)
            
    def test_experiment_directory_structure(self):
        """Test that the experiment creates the correct directory structure."""
        # Run experiment
        self.experiment.run()
        
        # Check experiment root directory
        exp_dir = os.path.join(self.workspace, "test_mnist")
        self.assertTrue(os.path.exists(exp_dir))
        
        # Check standard directories
        self.assertTrue(os.path.exists(os.path.join(exp_dir, "artifacts")))
        self.assertTrue(os.path.exists(os.path.join(exp_dir, "configs")))
        self.assertTrue(os.path.exists(os.path.join(exp_dir, "logs")))
        self.assertTrue(os.path.exists(os.path.join(exp_dir, "trials")))
        
        # Check config files
        self.assertTrue(os.path.exists(os.path.join(exp_dir, "configs", "experiment.yaml")))
        self.assertTrue(os.path.exists(os.path.join(exp_dir, "configs", "env.yaml")))
        
        # Check trials directory structure
        trials_dir = os.path.join(exp_dir, "trials")
        trial_names = ["small_perceptron", "medium_perceptron", "large_perceptron"]
        
        for trial_name in trial_names:
            trial_dir = os.path.join(trials_dir, trial_name)
            self.assertTrue(os.path.exists(trial_dir))
            
            # Check trial subdirectories
            self.assertTrue(os.path.exists(os.path.join(trial_dir, "artifacts")))
            self.assertTrue(os.path.exists(os.path.join(trial_dir, "configs")))
            self.assertTrue(os.path.exists(os.path.join(trial_dir, "logs")))
            
            # Check trial config files
            self.assertTrue(os.path.exists(os.path.join(trial_dir, "configs", "env.yaml")))
            self.assertTrue(os.path.exists(os.path.join(trial_dir, "configs", "trial.yaml")))
            
            # Check trial run directory
            trial_run_dir = os.path.join(trial_dir, trial_name)
            self.assertTrue(os.path.exists(trial_run_dir))
            self.assertTrue(os.path.exists(os.path.join(trial_run_dir, "artifacts")))
            self.assertTrue(os.path.exists(os.path.join(trial_run_dir, "configs")))
            self.assertTrue(os.path.exists(os.path.join(trial_run_dir, "logs")))
            self.assertTrue(os.path.exists(os.path.join(trial_run_dir, "outputs")))
            
    def test_log_files(self):
        """Test that log files are created and contain expected content."""
        # Run experiment
        self.experiment.run()
        
        # Get experiment log directory
        exp_dir = os.path.join(self.workspace, "test_mnist")
        log_dir = os.path.join(exp_dir, "logs")
        
        # Check that log file exists
        log_files = os.listdir(log_dir)
        self.assertEqual(len(log_files), 1)
        log_file = log_files[0]
        
        # Check log file name format (log_YYYYMMDD_HHMMSS.log)
        self.assertTrue(log_file.startswith("log_"))
        self.assertTrue(log_file.endswith(".log"))
        timestamp_str = log_file[4:-4]  # Extract YYYYMMDD_HHMMSS
        try:
            datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError:
            self.fail("Log file name does not match expected format")
            
        # Check log file content
        with open(os.path.join(log_dir, log_file), 'r') as f:
            content = f.read()
            
        # Check for expected log messages
        self.assertIn("Setting up experiment configuration", content)
        self.assertIn("Experiment setup complete", content)
        
        # Check trial logs
        trials_dir = os.path.join(exp_dir, "trials")
        trial_names = ["small_perceptron", "medium_perceptron", "large_perceptron"]
        
        for trial_name in trial_names:
            trial_dir = os.path.join(trials_dir, trial_name)
            trial_log_dir = os.path.join(trial_dir, "logs")
            
            # Check that trial log file exists
            trial_log_files = os.listdir(trial_log_dir)
            self.assertEqual(len(trial_log_files), 1)
            trial_log_file = trial_log_files[0]
            
            # Check trial log content
            with open(os.path.join(trial_log_dir, trial_log_file), 'r') as f:
                content = f.read()
                
            # Check for expected trial log messages
            self.assertIn(f"Trial '{trial_name}'", content)
            self.assertIn("Using device: cpu", content)
            self.assertIn("Created network: SingleLayerPerceptron", content)
            self.assertIn("Starting pipeline execution", content)
            
            # Check trial run logs
            trial_run_dir = os.path.join(trial_dir, trial_name)
            trial_run_log_dir = os.path.join(trial_run_dir, "logs")
            
            # Check that trial run log file exists
            trial_run_log_files = os.listdir(trial_run_log_dir)
            self.assertEqual(len(trial_run_log_files), 1)
            
if __name__ == '__main__':
    unittest.main()
