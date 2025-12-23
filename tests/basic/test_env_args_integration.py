import os
import shutil
import yaml
import pytest
from experiment_manager.experiment import Experiment
from experiment_manager.common.factory_registry import FactoryRegistry, FactoryType
from tests.pipelines.dummy_pipeline_factory import DummyPipelineFactory

@pytest.fixture
def config_dir_with_env_args(tmp_path):
    # Copy the existing config directory to a temp location
    src = os.path.join("tests", "configs", "test_experiment")
    dst = tmp_path / "test_experiment_env_args"
    shutil.copytree(src, dst)
    
    # Set the workspace in the env.yaml file
    env_path = dst / "env.yaml"
    with open(env_path, "r") as f:
        env_config = yaml.safe_load(f)
    env_config["workspace"] = os.path.join(tmp_path, "test_outputs")
    with open(env_path, "w") as f:
        yaml.dump(env_config, f, default_flow_style=False)
    
    trials_path = dst / "trials.yaml"
    # Load, modify, and save the trials config
    with open(trials_path, "r") as f:
        trials = yaml.safe_load(f)
    # Add args and switch pipeline to EnvArgsCheckPipeline for first trial
    trials[0]["args"] = {"custom_arg": 42}
    trials[0]["settings"]["pipeline"]["type"] = "EnvArgsCheckPipeline"
    trials[0]["settings"]["pipeline"]["assert_custom_arg"] = True
    trials[0]["settings"]["pipeline"]["expected_custom_arg"] = 42
    # Use the same pipeline for the second trial but no args
    trials[1]["settings"]["pipeline"]["type"] = "EnvArgsCheckPipeline"
    trials[1]["settings"]["pipeline"]["assert_custom_arg"] = True
    trials[1]["settings"]["pipeline"]["expected_custom_arg"] = None
    with open(trials_path, "w") as f:
        yaml.dump(trials, f, default_flow_style=False)
    # Print the written YAML for debugging
        # Assert loaded YAML is a list before writing
    assert isinstance(trials, list), f"Expected list, got {type(trials)} with value {trials}"
    with open(trials_path, "w") as f:
        yaml.dump(trials, f, default_flow_style=False)
    # Print and assert the written YAML is a list
    with open(trials_path, "r") as f:
        content = f.read()
        print("\n===== trials.yaml written by test =====\n")
        print(content)
        print("\n======================================\n")
        loaded = yaml.safe_load(content)
        assert isinstance(loaded, list), f"Written YAML is not a list: {loaded}"
    print(f"\nTemp config dir used by test: {dst}")
    print(f"Temp trials.yaml path: {trials_path}")
    return str(dst)

def test_env_args_reaches_pipeline(config_dir_with_env_args):
    import os
    print(f"\nListing files in test config dir: {config_dir_with_env_args}")
    for root, dirs, files in os.walk(config_dir_with_env_args):
        print(f"Dir: {root}")
        for f in files:
            print(f" - {f}")
    trials_path = os.path.join(config_dir_with_env_args, "trials.yaml")
    print(f"\nAbout to read temp trials.yaml at: {trials_path}")
    if not os.path.exists(trials_path):
        print("ERROR: trials.yaml does not exist!")
        assert False, f"trials.yaml missing at {trials_path}"
    with open(trials_path, "r") as f:
        content = f.read()
        print("\n===== trials.yaml before Experiment.create =====\n")
        print(content)
        print("\n==============================================\n")
    # Create custom factory registry
    registry = FactoryRegistry()
    registry.register(FactoryType.PIPELINE, DummyPipelineFactory())
    
    experiment = Experiment.create(config_dir_with_env_args, registry)
    experiment.run()
    # Check logs for evidence
    found_custom_arg_42 = False
    found_custom_arg_none = False
    for root, dirs, files in os.walk(experiment.env.workspace):
        for file in files:
            if "log" in file:
                file_path = os.path.join(root, file)
                try:
                    # Try to read as text file, skip if it's binary
                    with open(file_path, "r", encoding='utf-8') as f:
                        content = f.read()
                        if "custom_arg in env.args: 42" in content:
                            found_custom_arg_42 = True
                        if "custom_arg in env.args: None" in content:
                            found_custom_arg_none = True
                except (UnicodeDecodeError, UnicodeError):
                    # Skip binary files
                    print(f"Skipping binary file: {file_path}")
                    continue
    assert found_custom_arg_42, "custom_arg=42 not found in any pipeline log for trial 1"
    assert found_custom_arg_none, "custom_arg=None not found in any pipeline log for trial 2"
