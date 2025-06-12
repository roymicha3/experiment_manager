from omegaconf import DictConfig

from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.common.factory import Factory

# import all datasources
from experiment_manager.results.sources.filesystem_datasource import FileSystemDataSource
from experiment_manager.results.sources.db_datasource import DBDataSource


class DataSourceFactory(Factory):
    
    @staticmethod
    def create(name: str, config: DictConfig):
        class_ = YAMLSerializable.get_by_name(name)
        return class_.from_config(config)

