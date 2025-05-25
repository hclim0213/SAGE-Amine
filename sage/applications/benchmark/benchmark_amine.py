"""
Copyright (c) 2022 Hocheol Lim.
"""

import sys
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, Mol, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

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

def median_deae_eo_and_1m_2ppe(
    mean_cls=GeometricMeanScoringFunction,
) -> GoalDirectedBenchmark:
    t_target_1 = TanimotoScoringFunction("CCN(CC)CCOCCO", fp_type="ECFP6")
    t_target_2 = TanimotoScoringFunction("CN1CCCCC1CO", fp_type="ECFP6")
    median = mean_cls([t_target_1, t_target_2])

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(
        name="Median molecules 1",
        objective=median,
        contribution_specification=specification,
    )

def median_deae_eo_and_1_2he_pp(
    mean_cls=GeometricMeanScoringFunction,
) -> GoalDirectedBenchmark:
    t_target_1 = TanimotoScoringFunction("CCN(CC)CCOCCO", fp_type="ECFP6")
    t_target_2 = TanimotoScoringFunction("OCCN1CCCCC1", fp_type="ECFP6")
    median = mean_cls([t_target_1, t_target_2])

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(
        name="Median molecules 2",
        objective=median,
        contribution_specification=specification,
    )

def median_1m_2ppe_and_1_2he_pp(
    mean_cls=GeometricMeanScoringFunction,
) -> GoalDirectedBenchmark:
    t_target_1 = TanimotoScoringFunction("CN1CCCCC1CO", fp_type="ECFP6")
    t_target_2 = TanimotoScoringFunction("OCCN1CCCCC1", fp_type="ECFP6")
    median = mean_cls([t_target_1, t_target_2])

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(
        name="Median molecules 3",
        objective=median,
        contribution_specification=specification,
    )

def isomers_c4h11no(mean_function="geometric", n_samples=56) -> GoalDirectedBenchmark:

    specification = uniform_specification(n_samples)

    return GoalDirectedBenchmark(
        name="C4H11NO",
        objective=IsomerScoringFunction("C4H11NO", mean_function=mean_function),
        contribution_specification=specification,
    )

def isomers_c4h11no2(mean_function="geometric", n_samples=284) -> GoalDirectedBenchmark:

    specification = uniform_specification(n_samples)

    return GoalDirectedBenchmark(
        name="C4H11NO2",
        objective=IsomerScoringFunction("C4H11NO2", mean_function=mean_function),
        contribution_specification=specification,
    )

def isomers_c5h12n2(mean_function="geometric", n_samples=716) -> GoalDirectedBenchmark:

    specification = uniform_specification(n_samples)

    return GoalDirectedBenchmark(
        name="C5H12N2",
        objective=IsomerScoringFunction("C5H12N2", mean_function=mean_function),
        contribution_specification=specification,
    )


def isomers_c6h15no(mean_function="geometric", n_samples=398) -> GoalDirectedBenchmark:

    specification = uniform_specification(n_samples)

    return GoalDirectedBenchmark(
        name="C6H15NO",
        objective=IsomerScoringFunction("C6H15NO", mean_function=mean_function),
        contribution_specification=specification,
    )

def similarity(
    smiles: str,
    name: str,
    fp_type: str = "ECFP6",
    threshold: float = 0.75,
    rediscovery: bool = False,
) -> GoalDirectedBenchmark:
    category = "rediscovery" if rediscovery else "similarity"
    benchmark_name = f"{name} {category}"

    modifier = ClippedScoreModifier(upper_x=threshold)
    scoring_function = TanimotoScoringFunction(
        target=smiles, fp_type=fp_type, score_modifier=modifier
    )
    if rediscovery:
        specification = uniform_specification(1)
    else:
        specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(
        name=benchmark_name,
        objective=scoring_function,
        contribution_specification=specification,
    )
