from abc import ABC, abstractmethod
from experiment_manager.results.sources.datasource import ExperimentDataSource


class Extractor(ABC):
    @abstractmethod
    def extract(self, datasource: ExperimentDataSource):
        pass