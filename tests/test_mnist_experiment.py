import os
import unittest
from datetime import datetime
from omegaconf import OmegaConf

from experiment_manager.environment import Environment
from experiment_manager.experiment import Experiment
from examples.pipelines.pipeline_factory_example import ExamplePipelineFactory

class TestMNISTExperiment(unittest.TestCase):
    def setUp(self):
        # Create a temporary workspace for testing
        self.workspace = os.path.abspath("outputs/test_workspace")
        os.makedirs(self.workspace, exist_ok=True)
        
        # Create factory
        self.factory = ExamplePipelineFactory
        
        # Create environment
        self.env = Environment(
            workspace=self.workspace,
            config=OmegaConf.create({
                "settings": {
                    "device": "cpu",
                    "debug": True,
                    "verbose": True
                }
            }),
            factory=self.factory
        )
        self.env.setup_environment()
        
        # Load experiment config
        self.config = OmegaConf.create({
            "name": "test_mnist",
            "id": 1,
            "desc": "Test MNIST experiment",
            "config_dir_path": os.path.join(os.path.dirname(__file__), "..", "examples", "mnist_experiment", "configs"),
            "settings": {
                "device": "cpu",
                "debug": True,
                "verbose": True
            }
        })
        
        # Create experiment
        self.experiment = Experiment.from_config(self.config, self.env)
        
        # Create trial configs
        self.experiment.base_config = OmegaConf.create({
            "settings": {
                "device": "cpu",
                "debug": True,
                "verbose": True
            },
            "pipeline": {
                "type": "TrainingPipeline"
            },
            "training": {
                "dataset": "mnist",
                "validation_split": 0.1,
                "test_split": 0.1,
                "batch_size": 64,
                "shuffle": True,
                "epochs": 10,
                "optimizer": "adam"
            }
        })
        
        self.experiment.trials_config = [
            OmegaConf.create({
                "name": "small_perceptron",
                "id": 1,
                "repeat": 1,
                "settings": {
                    "device": "cpu",
                    "debug": True,
                    "verbose": True
                },
                "pipeline": {
                    "type": "TrainingPipeline"
                },
                "training": {
                    "input_size": 784,
                    "hidden_size": 128,
                    "num_classes": 10,
                    "learning_rate": 0.001,
                    "dataset": "mnist",
                    "validation_split": 0.1,
                    "test_split": 0.1,
                    "batch_size": 64,
                    "shuffle": True,
                    "epochs": 10,
                    "optimizer": "adam"
                }
            }),
            OmegaConf.create({
                "name": "medium_perceptron",
                "id": 2,
                "repeat": 1,
                "settings": {
                    "device": "cpu",
                    "debug": True,
                    "verbose": True
                },
                "pipeline": {
                    "type": "TrainingPipeline"
                },
                "training": {
                    "input_size": 784,
                    "hidden_size": 256,
                    "num_classes": 10,
                    "learning_rate": 0.001,
                    "dataset": "mnist",
                    "validation_split": 0.1,
                    "test_split": 0.1,
                    "batch_size": 64,
                    "shuffle": True,
                    "epochs": 10,
                    "optimizer": "adam"
                }
            }),
            OmegaConf.create({
                "name": "large_perceptron",
                "id": 3,
                "repeat": 1,
                "settings": {
                    "device": "cpu",
                    "debug": True,
                    "verbose": True
                },
                "pipeline": {
                    "type": "TrainingPipeline"
                },
                "training": {
                    "input_size": 784,
                    "hidden_size": 512,
                    "num_classes": 10,
                    "learning_rate": 0.001,
                    "dataset": "mnist",
                    "validation_split": 0.1,
                    "test_split": 0.1,
                    "batch_size": 64,
                    "shuffle": True,
                    "epochs": 2,
                    "optimizer": "adam"
                }
            })
        ]
        
    def tearDown(self):
        # Clean up any open file handles
        if hasattr(self, 'experiment') and hasattr(self.experiment, 'env'):
            if hasattr(self.experiment.env, 'trials_env'):
                for trial in getattr(self.experiment.env.trials_env, 'trials', []):
                    if hasattr(trial, 'logger'):
                        trial.logger = None
            self.experiment.env.logger = None
        if hasattr(self, 'env'):
            self.env.logger = None
        
        # Wait a bit to ensure all files are released
        import time
        time.sleep(1)
        
        # Clean up test workspace
        import shutil
        if os.path.exists(self.workspace):
            try:
                shutil.rmtree(self.workspace)
            except (PermissionError, OSError):
                # If files are still locked, wait a bit more and try again
                time.sleep(2)
                try:
                    shutil.rmtree(self.workspace)
                except (PermissionError, OSError):
                    # If still can't delete, just skip cleanup
                    pass
            
    def test_experiment_directory_structure(self):
        """Test that the experiment creates the correct directory structure."""
        # Run experiment
        self.experiment.run()
        
        # Wait a bit to ensure all files are written
        import time
        time.sleep(1)
        
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
            
    
            
if __name__ == '__main__':
    unittest.main()
