from experiment_manager.common.serializable import YAMLSerializable
from .simple_classifier import SimpleClassifierPipeline

@YAMLSerializable.register('SimpleClassifierPipeline')
class _SimpleClassifierPipeline(SimpleClassifierPipeline):
    pass
