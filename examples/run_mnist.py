import os
import torch
from omegaconf import DictConfig, OmegaConf

from experiment_manager.environment import Environment
from pipeline_example import TrainingPipeline

def main():
    # Create environment
    workspace = os.path.join(os.path.dirname(__file__), "mnist_workspace")
    config = DictConfig({
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })
    env = Environment(workspace, config, verbose=True)
    env.setup_environment()
    
    # Create pipeline config
    pipeline_config = DictConfig({
        'epochs': 10,
        'batch_size': 64,
        'validation_split': 0.1,
        'test_split': 0.1,
        'shuffle': True,
        'callbacks': [
            {
                'type': 'EarlyStopping',
                'metric': 'val_loss',
                'patience': 3,
                'min_delta_percent': 0.1
            },
            {
                'type': 'MetricsTracker'
            }
        ]
    })
    
    # Create and run pipeline
    pipeline = TrainingPipeline.from_config(pipeline_config, env)
    pipeline.run(pipeline_config)

if __name__ == "__main__":
    main()
