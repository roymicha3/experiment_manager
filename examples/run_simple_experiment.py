import os
from omegaconf import OmegaConf

from experiment_manager.environment import Environment
from experiment_manager.experiment import Experiment
from examples.pipelines.dummy_pipeline_factory import DummyPipelineFactory

def main():
    # Create environment config
    env_config = OmegaConf.create({
        "workspace": os.path.join("outputs", "simple_experiment"),
        "verbose": True,
        "debug": True,
        "trackers": [
            {
                "type": "LogTracker",
                "name": "LogTracker",
                "verbose": True
            }
        ]
    })

    # Initialize environment
    env = Environment.from_config(env_config)
    env.setup_environment()
    print(f"Environment workspace: {env.workspace}")
    print(f"Environment log directory: {env.log_dir}")

    # Create and run experiment
    config_dir = os.path.join(os.path.dirname(__file__), "configs", "simple_experiment")
    experiment = Experiment.create(config_dir, DummyPipelineFactory)
    print(f"\nCreated experiment: {experiment.name}")
    print(f"Experiment workspace: {experiment.env.workspace}")
    
    # Run the experiment
    print("\nRunning experiment...")
    experiment.run()
    print("Experiment completed!")

    # Print directory structure
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(experiment.env.workspace):
        level = root.replace(experiment.env.workspace, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

if __name__ == "__main__":
    main()
