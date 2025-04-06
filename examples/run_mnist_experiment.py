import os
import torch
from omegaconf import OmegaConf, DictConfig

from experiment_manager.environment import Environment
from experiment_manager.experiment import Experiment
from examples.pipelines.pipeline_factory_example import ExamplePipelineFactory

def main():
    # Create environment configuration
    env_config = OmegaConf.create({
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "settings": {
            "debug": True,
            "verbose": True
        }
    })
    
    # Create and setup environment with pipeline factory
    workspace = os.path.abspath("outputs")
    env = Environment(
        workspace=workspace, 
        config=env_config,
        factory=ExamplePipelineFactory,
        verbose=True
    )
    env.setup_environment()
    
    # Load experiment configuration
    exp_config_path = os.path.join(
        os.path.dirname(__file__), 
        "mnist_experiment/configs/experiment.yaml"
    )
    exp_config = OmegaConf.load(exp_config_path)
    
    # Create and run experiment
    experiment = Experiment.from_config(exp_config, env)
    experiment.run()
    
    # Print experiment structure
    print("\nExperiment structure:")
    for root, dirs, files in os.walk(experiment.env.workspace):
        level = root.replace(experiment.env.workspace, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files:
            print(f"{indent}    {f}")

if __name__ == "__main__":
    main()
