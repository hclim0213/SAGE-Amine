"""
Copyright (c) 2024 Hocheol Lim.
"""

import os
import sys

import networkx as nx
from sage.scoring.common_scoring_functions import (
    CNS_MPO_ScoringFunction,
    IsomerScoringFunction,
    RdkitScoringFunction,
    SMARTSScoringFunction,
    TanimotoScoringFunction,
)
from sage.scoring.goal_directed_benchmark import GoalDirectedBenchmark
from sage.scoring.goal_directed_score_contributions import uniform_specification
from sage.scoring.score_modifier import (
    ClippedScoreModifier,
    GaussianModifier,
    MaxGaussianModifier,
    MinGaussianModifier,
)
from sage.scoring.scoring_function import (
    ArithmeticMeanScoringFunction,
    GeometricMeanScoringFunction,
    ScoringFunction,
    MoleculewiseScoringFunction,
)

from sage.scoring.amine.descriptors_amine import (
    high_pka_ps_score,
    high_pka_tcm_score,
    high_pka_score,
    low_viscosity_ps_score,
    low_viscosity_tcm_score,
    low_viscosity_score,
    low_vapor_pressure_ps_score,
    low_vapor_pressure_tcm_score,
    low_vapor_pressure_score,
    high_co2_absorption_ps_score,
    high_co2_absorption_tcm_score,
    high_co2_absorption_score,
    high_co2_absorption_pst_score,
    high_co2_absorption_all_score,
)

from rdkit import Chem
from rdkit.Chem import Descriptors, Mol

class ThresholdedImprovementScoringFunction(MoleculewiseScoringFunction):
    def __init__(self, objective, constraint, threshold, offset):
        super().__init__()
        self.objective = objective
        self.constraint = constraint
        self.threshold = threshold
        self.offset = offset

    def raw_score(self, smiles):
        score = (
            self.corrupt_score
            if (self.constraint.score(smiles) < self.threshold)
            else (self.objective.score(smiles) + self.offset)
        )
        return score

def high_pka_ps_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="High pKa with Primary and Secondary Amines",
        objective=RdkitScoringFunction(descriptor=high_pka_ps_score),
        contribution_specification=specification,
    )

def high_pka_tcm_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="High pKa with Tertiary, Cyclic, and Multi-Amines",
        objective=RdkitScoringFunction(descriptor=high_pka_tcm_score),
        contribution_specification=specification,
    )

def high_pka_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="High pKa without Constraint",
        objective=RdkitScoringFunction(descriptor=high_pka_score),
        contribution_specification=specification,
    )

def low_viscosity_ps_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low Viscosity with Primary and Secondary Amines",
        objective=RdkitScoringFunction(descriptor=low_viscosity_ps_score),
        contribution_specification=specification,
    )

def low_viscosity_tcm_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low Viscosity with Tertiary, Cyclic, and Multi-Amines",
        objective=RdkitScoringFunction(descriptor=low_viscosity_tcm_score),
        contribution_specification=specification,
    )

def low_viscosity_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low Viscosity without Constraint",
        objective=RdkitScoringFunction(descriptor=low_viscosity_score),
        contribution_specification=specification,
    )

def low_vapor_pressure_ps_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low Vapor Pressure with Primary and Secondary Amines",
        objective=RdkitScoringFunction(descriptor=low_vapor_pressure_ps_score),
        contribution_specification=specification,
    )

def low_vapor_pressure_tcm_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low Vapor Pressure with Tertiary, Cyclic, and Multi-Amines",
        objective=RdkitScoringFunction(descriptor=low_vapor_pressure_tcm_score),
        contribution_specification=specification,
    )

def low_vapor_pressure_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="Low Vapor Pressure without Constraint",
        objective=RdkitScoringFunction(descriptor=low_vapor_pressure_score),
        contribution_specification=specification,
    )

def high_co2_absorption_ps_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="High CO2 Absorption with Primary and Secondary Amines",
        objective=RdkitScoringFunction(descriptor=high_co2_absorption_ps_score),
        contribution_specification=specification,
    )

def high_co2_absorption_tcm_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="High CO2 Absorption with Tertiary, Cyclic, and Multi-Amines",
        objective=RdkitScoringFunction(descriptor=high_co2_absorption_tcm_score),
        contribution_specification=specification,
    )

def high_co2_absorption_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="High CO2 Absorption without Constraint",
        objective=RdkitScoringFunction(descriptor=high_co2_absorption_score),
        contribution_specification=specification,
    )

def high_co2_absorption_pst_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="High CO2 Absorption with Primary, Secondary, and Tertiary Amines",
        objective=RdkitScoringFunction(descriptor=high_co2_absorption_pst_score),
        contribution_specification=specification,
    )

def high_co2_absorption_all_task() -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name="High CO2 Absorption without Constraint",
        objective=RdkitScoringFunction(descriptor=high_co2_absorption_all_score),
        contribution_specification=specification,
    )
