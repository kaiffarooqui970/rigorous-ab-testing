"""Simulation engine with known ground truth, plus a Monte Carlo validation harness."""

from xpkit.sim.harness import SimulationDraw, ValidationResult, results_frame, run_validation
from xpkit.sim.panel import PanelSpec, SimulatedPanel, simulate_panel
from xpkit.sim.simulate import (
    ExperimentSpec,
    SimulatedExperiment,
    population_truth,
    simulate_experiment,
)

__all__ = [
    "ExperimentSpec",
    "PanelSpec",
    "SimulatedExperiment",
    "SimulatedPanel",
    "SimulationDraw",
    "ValidationResult",
    "population_truth",
    "results_frame",
    "run_validation",
    "simulate_experiment",
    "simulate_panel",
]
