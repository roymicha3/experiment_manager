import os
import torch
from omegaconf import DictConfig, OmegaConf

from experiment_manager.environment import Environment
from examples.pipelines.pipeline_example import TrainingPipeline
from pipelines.pipeline_factory_example import ExamplePipelineFactory

def main():
    # Create environment with factory
    workspace = os.path.join(os.path.dirname(__file__), "mnist_workspace")
    config = DictConfig({
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })
    factory = ExamplePipelineFactory
    env = Environment(workspace, config, factory=factory, verbose=True)
    env.setup_environment()
    
    # Create pipeline config
    pipeline_config = DictConfig({
        'type': 'TrainingPipeline',  # Must match the registered name
        'epochs': 10,
        'batch_size': 64,
        'validation_split': 0.1,
        'test_split': 0.1,
        'shuffle': True,
        'input_size': 784,
        'num_classes': 10,
        'learning_rate': 0.001,
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
    
    # Create and run pipeline using factory
    pipeline = factory.create(pipeline_config.type, pipeline_config, env, id=1)
    pipeline.run(pipeline_config)

if __name__ == "__main__":
    main()
