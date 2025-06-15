# models/experiment.py

from dataclasses import dataclass
from typing import Dict, Optional, Union

from experiment_manager.common.common import ArtifactType, RunStatus

@dataclass
class Artifact:
    id: int
    type: ArtifactType | str  # Allow raw string for backward compatibility
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
    status: Union[RunStatus, str]
    num_epochs: int
    dir_path: Optional[str] = None

@dataclass
class Trial:
    id: int
    name: str
    experiment_id: int
    dir_path: Optional[str] = None

@dataclass
class Experiment:
    id: int
    name: str
    description: Optional[str] = None
    dir_path: Optional[str] = None
