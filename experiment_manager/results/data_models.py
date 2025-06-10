# models/experiment.py

from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Artifact:
    id: int
    type: str
    path: str

@dataclass
class MetricRecord:
    trial_run_id: int
    epoch: Optional[int]
    metrics: Dict[str, float]
    is_custom: bool = False

@dataclass
class TrialRun:
    id: int
    trial_id: int
    status: str
    metrics: List[MetricRecord]
    artifacts: List[Artifact]
    num_epochs: int
    dir_path: Optional[str] = None

@dataclass
class Trial:
    id: int
    name: str
    experiment_id: int
    runs: List[TrialRun]
    dir_path: Optional[str] = None

@dataclass
class Experiment:
    id: int
    name: str
    trials: List[Trial]
    description: Optional[str] = None
    dir_path: Optional[str] = None
