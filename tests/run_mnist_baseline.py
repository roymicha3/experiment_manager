"""
Test runner for MNIST baseline experiment.
This generates real experiment data that can be used in our tests.
"""
import os
import sys
import tempfile
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiment_manager.experiment import Experiment
from tests.pipelines.test_pipeline_factory import TestPipelineFactory


def run_mnist_baseline_experiment(workspace_dir=None):
    """
    Run the MNIST baseline experiment and return the experiment object.
    
    Args:
        workspace_dir: Directory to use for experiment workspace. 
                      If None, uses the config default.
    
    Returns:
        experiment: The completed experiment object
        db_path: Path to the experiment database
    """
    # Get config directory
    config_dir = os.path.join(os.path.dirname(__file__), "configs", "test_mnist_baseline")
    
    # Update workspace if provided
    if workspace_dir:
        from omegaconf import OmegaConf
        env_path = os.path.join(config_dir, "env.yaml")
        env_config = OmegaConf.load(env_path)
        env_config.workspace = workspace_dir
        OmegaConf.save(env_config, env_path)
    
    # Create and run experiment
    experiment = Experiment.create(config_dir, TestPipelineFactory)
    print(f"Created experiment: {experiment.name}")
    print(f"Experiment workspace: {experiment.env.workspace}")
    
    # Run the experiment
    print("Running MNIST baseline experiment...")
    experiment.run()
    print("Experiment completed!")
    
    # Get database path
    db_path = os.path.join(experiment.env.artifact_dir, "experiment.db")
    
    return experiment, db_path


def main():
    """Run the baseline experiment for testing."""
    try:
        experiment, db_path = run_mnist_baseline_experiment()
        
        print(f"\nExperiment completed successfully!")
        print(f"Database path: {db_path}")
        print(f"Workspace: {experiment.env.workspace}")
        
        # Check database directly first
        import sqlite3
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM EXPERIMENT")
            exp_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM TRIAL")
            trial_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM TRIAL_RUN")
            run_count = cursor.fetchone()[0]
            conn.close()
            
            print(f"\nDirect database check:")
            print(f"- Experiments: {exp_count}")
            print(f"- Trials: {trial_count}")
            print(f"- Trial runs: {run_count}")
            
            if exp_count == 0:
                print("\nWarning: No data was saved to database!")
                print("This suggests an issue with the DBTracker integration.")
                return experiment, db_path
        except Exception as e:
            print(f"Error checking database directly: {e}")
        
        # Get experiment data from database
        from experiment_manager.results.sources.db_datasource import DBDataSource
        
        with DBDataSource(db_path) as source:
            exp_data = source.get_experiment()
            
            # Print basic stats
            print(f"\nExperiment stats:")
            print(f"- Trials: {len(exp_data.trials)}")
            total_runs = sum(len(trial.runs) for trial in exp_data.trials)
            print(f"- Total runs: {total_runs}")
            
            # Print trial details
            for trial in exp_data.trials:
                print(f"  - Trial '{trial.name}': {len(trial.runs)} runs")
        
        # Print directory structure
        print("\nWorkspace structure:")
        workspace = Path(experiment.env.workspace)
        for root, dirs, files in os.walk(workspace):
            level = root.replace(str(workspace), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for f in files:
                print(f"{subindent}{f}")
                
    except Exception as e:
        print(f"Error running experiment: {e}")
        raise


if __name__ == "__main__":
    main() 
