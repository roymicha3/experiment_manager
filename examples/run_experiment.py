import os
from omegaconf import OmegaConf
from experiment_manager.environment import Environment
from experiment_manager.experiment import Experiment


def main():
    # Create environment configuration
    env_config = OmegaConf.create({
        "workspace": "outputs",  # All experiment outputs will be in this directory
        "settings": {
            "debug": True,
            "verbose": True
        }
    })
    
    # Create and setup environment
    workspace = os.path.abspath("outputs")
    env = Environment(workspace=workspace, config=env_config)
    env.setup_environment()
    
    # Create experiment configuration
    exp_config = OmegaConf.create({
        "name": "hyperparameter_search",
        "id": 1,
        "desc": "Training a model with different hyperparameters",
        "config_dir_path": os.path.abspath("examples/configs")  # Path to config files
    })
    
    # Create and run experiment
    experiment = Experiment.from_config(exp_config, env)
    
    # Run all trials
    experiment.run()
    
    # The experiment will:
    # 1. Create a workspace at outputs/hyperparameter_search/
    # 2. For each trial, create a workspace at outputs/hyperparameter_search/trials/{trial_name}/
    # 3. Run each trial the specified number of times
    # 4. Save logs and configs in their respective directories
    
    print("\nExperiment structure:")
    for root, dirs, files in os.walk(experiment.env.workspace):
        level = root.replace(experiment.env.workspace, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files:
            print(f"{indent}    {f}")


if __name__ == "__main__":
    main()
