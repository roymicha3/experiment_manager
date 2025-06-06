from experiment_manager.common.serializable import YAMLSerializable
from tests.pipelines.simple_classifier import SimpleClassifierPipeline

@YAMLSerializable.register('SimpleClassifierPipeline')
class _SimpleClassifierPipeline(SimpleClassifierPipeline):
    pass
