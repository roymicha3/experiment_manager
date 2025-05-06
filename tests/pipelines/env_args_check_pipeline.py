from omegaconf import DictConfig
from experiment_manager.environment import Environment
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.common.serializable import YAMLSerializable

@YAMLSerializable.register("EnvArgsCheckPipeline")
class EnvArgsCheckPipeline(Pipeline, YAMLSerializable):
    """
    Pipeline that asserts the presence and value of custom env args for testing.
    """
    def __init__(self, env: Environment, id: int = None):
        Pipeline.__init__(self, env)
        YAMLSerializable.__init__(self)
        self.name = "EnvArgsCheckPipeline"

    @classmethod
    def from_config(cls, config: DictConfig, env: Environment, id: int = None):
        return cls(env, id)

    def run(self, config: DictConfig):
        # Check for a specific custom_arg in env.args
        custom_arg = self.env.args.get("custom_arg", None)
        self.env.logger.info(f"[EnvArgsCheckPipeline] custom_arg in env.args: {custom_arg}")
        # Optionally, assert for test failure if required
        if config.get("assert_custom_arg", False):
            assert custom_arg == config.get("expected_custom_arg"), (
                f"custom_arg was {custom_arg}, expected {config.get('expected_custom_arg')}")
        return custom_arg
