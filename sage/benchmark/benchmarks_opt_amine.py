"""
Copyright (c) 2024 Hocheol Lim.
"""
import sys
sys.path.append('/workspace/_ext')

from typing import List, Tuple
from sage.scoring.goal_directed_benchmark import GoalDirectedBenchmark

from sage.applications.benchmark.benchmark_amine import (
    similarity,
    median_deae_eo_and_1m_2ppe,
    median_deae_eo_and_1_2he_pp,
    median_1m_2ppe_and_1_2he_pp,
    isomers_c4h11no,
    isomers_c4h11no2,
    isomers_c5h12n2,
    isomers_c6h15no,
)

from sage.applications.amine.co2_absorption import (
    high_pka_ps_task,
    high_pka_tcm_task,
    high_pka_task,
    low_viscosity_ps_task,
    low_viscosity_tcm_task,
    low_viscosity_task,
    low_vapor_pressure_ps_task,
    low_vapor_pressure_tcm_task,
    low_vapor_pressure_task,
    high_co2_absorption_ps_task,
    high_co2_absorption_tcm_task,
    high_co2_absorption_task,
    high_co2_absorption_pst_task,
    high_co2_absorption_all_task,
)

#   Amine Benchmarks

#   00 : MEA Rediscovery
#   01 : AHMPD Rediscovery
#   02 : DEA Rediscovery
#   03 : DA Rediscovery
#   04 : DEEA Rediscovery
#   05 : MDEA Rediscovery
#   06 : 2-MPZ Rediscovery
#   07 : 2-PPE Rediscovery
#   08 : HomoPZ Rediscovery
#   09 : DETA Rediscovery

#   10 : AMP Similarity
#   11 : IPA Similarity
#   12 : IPAP Similarity
#   13 : 4DMA1B Similarity
#   14 : 1DMA2P Similarity
#   15 : 1-MPZ Similarity
#   16 : EPZ Similarity
#   17 : PZ Similarity
#   18 : AEP Similarity
#   19 : TETA Similarity

#   20 : Median Similarity of DEAE-EO and 1M-2PPE
#   21 : Median Similarity of DEAE-EO and 1-(2HE)PP
#   22 : Median Similarity of 1M-2PPE and 1-(2HE)PP

#   23 : Isomers C4H11NO
#   24 : Isomers C4H11NO2
#   25 : Isomers C5H12N2
#   26 : Isomers C6H15NO

#   Amine Single Property Optimization

#   27 : High pKa (Primary and Secondary Amines)
#   28 : High pKa (Tertiary, Cyclic, and Multi-Amines)
#   29 : High pKa (No constraint)

#   30 : Low Viscosity (Primary and Secondary Amines)
#   31 : Low Viscosity (Tertiary, Cyclic, and Multi-Amines)
#   32 : Low Viscosity (No constraint)

#   33 : Low Vapor Pressure (Primary and Secondary Amines)
#   34 : Low Vapor Pressure (Tertiary, Cyclic, and Multi-Amines)
#   35 : Low Vapor Pressure (No constraint)

#   Amine Multiple Property Optimization

#   36 : High CO2 Absorption (Primary and Secondary Amines)
#   37 : High CO2 Absorption (Tertiary, Cyclic, and Multi-Amines)
#   38 : High CO2 Absorption (No constraint)

#   Amine MPO Addition
#   39 : MW200 + High CO2 Absorption (Primary, Secondary, and Tertiary Amines)
#   40 : MW200 + High CO2 Absorption (No constraint)

def load_benchmark(benchmark_id: int) -> Tuple[GoalDirectedBenchmark, List[int]]:
    benchmark = {
        
        # Amine Benchmark
        
        0: similarity(
            smiles="NCCO",
            name="MEA",
            fp_type="ECFP6",
            threshold=1.0,
            rediscovery=True,
        ),
        1: similarity(
            smiles="NC(CO)(CO)CO",
            name="AHMPD",
            fp_type="ECFP6",
            threshold=1.0,
            rediscovery=True,
        ),
        2: similarity(
            smiles="OCCNCCO",
            name="DEA",
            fp_type="ECFP6",
            threshold=1.0,
            rediscovery=True,
        ),
        3: similarity(
            smiles="CCNCC",
            name="DA",
            fp_type="ECFP6",
            threshold=1.0,
            rediscovery=True,
        ),
        4: similarity(
            smiles="CCN(CC)CCO",
            name="DEEA",
            fp_type="ECFP6",
            threshold=1.0,
            rediscovery=True,
        ),
        5: similarity(
            smiles="CN(CCO)CCO",
            name="MDEA",
            fp_type="ECFP6",
            threshold=1.0,
            rediscovery=True,
        ),
        6: similarity(
            smiles="CC1CNCCN1",
            name="2-MPZ",
            fp_type="ECFP6",
            threshold=1.0,
            rediscovery=True,
        ),
        7: similarity(
            smiles="OCCC1CCCCN1",
            name="2-PPE",
            fp_type="ECFP6",
            threshold=1.0,
            rediscovery=True,
        ),
        8: similarity(
            smiles="C1CNCCNC1",
            name="HomoPZ",
            fp_type="ECFP6",
            threshold=1.0,
            rediscovery=True,
        ),
        9: similarity(
            smiles="NCCNCCN",
            name="DETA",
            fp_type="ECFP6",
            threshold=1.0,
            rediscovery=True,
        ),
        10: similarity(
            smiles="CC(C)(N)CO",
            name="AMP",
            fp_type="ECFP6",
            threshold=0.75,
        ),
        11: similarity(
            smiles="CC(C)N",
            name="IPA",
            fp_type="ECFP6",
            threshold=0.75,
        ),
        12: similarity(
            smiles="CC(C)NCCCO",
            name="IPAP",
            fp_type="ECFP6",
            threshold=0.75,
        ),
        13: similarity(
            smiles="CN(C)CCCCO",
            name="4DMA1B",
            fp_type="ECFP6",
            threshold=0.75,
        ),
        14: similarity(
            smiles="CC(O)CN(C)C",
            name="1DMA2P",
            fp_type="ECFP6",
            threshold=0.75,
        ),
        15: similarity(
            smiles="CN1CCNCC1",
            name="1-MPZ",
            fp_type="ECFP6",
            threshold=0.75,
        ),
        16: similarity(
            smiles="CCN1CCNCC1",
            name="EPZ",
            fp_type="ECFP6",
            threshold=0.75,
        ),
        17: similarity(
            smiles="C1CNCCN1",
            name="PZ",
            fp_type="ECFP6",
            threshold=0.75,
        ),
        18: similarity(
            smiles="NCCN1CCNCC1",
            name="AEP",
            fp_type="ECFP6",
            threshold=0.75,
        ),
        19: similarity(
            smiles="NCCNCCNCCN",
            name="TETA",
            fp_type="ECFP6",
            threshold=0.75,
        ),
        20: median_deae_eo_and_1m_2ppe(),
        21: median_deae_eo_and_1_2he_pp(),
        22: median_1m_2ppe_and_1_2he_pp(),
        
        23: isomers_c4h11no(),
        24: isomers_c4h11no2(),
        25: isomers_c5h12n2(),
        26: isomers_c6h15no(),
        
        27: high_pka_ps_task(),
        28: high_pka_tcm_task(),
        29: high_pka_task(),
        30: low_viscosity_ps_task(),
        31: low_viscosity_tcm_task(),
        32: low_viscosity_task(),
        33: low_vapor_pressure_ps_task(),
        34: low_vapor_pressure_tcm_task(),
        35: low_vapor_pressure_task(),
        36: high_co2_absorption_ps_task(),
        37: high_co2_absorption_tcm_task(),
        38: high_co2_absorption_task(),
        39: high_co2_absorption_pst_task(),
        40: high_co2_absorption_all_task(),
        
    }.get(benchmark_id)

    if benchmark_id in [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ]:
        scoring_num_list = [1]
    elif benchmark_id in [23]:
        scoring_num_list = [56]
    elif benchmark_id in [24]:
        scoring_num_list = [284]
    elif benchmark_id in [25]:
        scoring_num_list = [716]
    elif benchmark_id in [26]:
        scoring_num_list = [398]
    else:
        scoring_num_list = [1, 10, 100]

    return benchmark, scoring_num_list  # type: ignore

