"""
Plot Plugins for Visualization Components

This package provides specialized plot implementations for common visualization
tasks in machine learning and experiment analysis workflows.
"""

from .training_curves import TrainingCurvesPlotPlugin

__all__ = [
    "TrainingCurvesPlotPlugin",
] 