"""Database tables for experiment tracking."""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

@dataclass
class Experiment:
    """Represents an experiment in the database."""
    id: Optional[int]
    title: str
    description: str
    start_time: datetime
    update_time: datetime

@dataclass
class Trial:
    """Represents a trial in the database."""
    id: Optional[int]
    name: str
    experiment_id: int
    start_time: datetime
    update_time: datetime

@dataclass
class TrialRun:
    """Represents a trial run in the database."""
    id: Optional[int]
    trial_id: int
    status: str
    start_time: datetime
    update_time: datetime

@dataclass
class Metric:
    """Represents a metric in the database."""
    id: Optional[int]
    type: str
    total_val: float
    per_label_val: Optional[Dict]

@dataclass
class Artifact:
    """Represents an artifact in the database."""
    id: Optional[int]
    type: str
    location: str
    
@dataclass
class Epoch:
    id: Optional[int]
    epoch_idx: int
    trial_run_id: int
    time: datetime
    
