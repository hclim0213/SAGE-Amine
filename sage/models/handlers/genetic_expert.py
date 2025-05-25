"""
Copyright (c) 2022 Hocheol Lim.
"""

import gc
import random
from typing import List

import numpy as np
import torch
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError
from rdkit import Chem

from sage.models.expert import (
    crossover,
    fragment_crossover,
    mutate,
    selfies_crossover,
    selfies_mutate,
)

class GeneticOperatorHandler:
    def __init__(
        self,
        crossover_type: str,
        mutation_type: str,
        mutation_initial_rate: float,
    ) -> None:
        self.mutation_initial_rate = mutation_initial_rate
        self.mutation_rate = mutation_initial_rate

        if crossover_type == "SMILES":
            self.crossover_func = crossover
        elif crossover_type == "SELFIES":
            self.crossover_func = selfies_crossover
        elif crossover_type == "ATTR":
            self.crossover_func = fragment_crossover  # type: ignore
        else:
            raise ValueError(f"'crossover_type' {crossover_type} is invalid")

        if mutation_type == "SMILES":
            self.mutate_func = mutate
        elif mutation_type == "SELFIES" or mutation_type == "ATTR":
            self.mutate_func = selfies_mutate
        else:
            raise ValueError(f"'mutation_type' {mutation_type} is invalid")

    def query(
        self,
        query_size: int,
        apprentice_mean_similarity: float,
        mating_pool: List[str],
        pool: Parallel,
    ) -> List[str]:

        if self.crossover_func.__name__ == "fragment_crossover":
            smiles = pool(
                delayed(self.reproduce_frags)(mating_pool, self.mutation_rate)
                for _ in range(query_size)
            )
        else:
            original_smiles = random.choices(mating_pool, k=2 * query_size)
            smiles_a, smiles_b = (
                original_smiles[:query_size],
                original_smiles[query_size:],
            )
            
            def reproduce_mols_wt_timeout(smile_a, smile_b, mutation_rate, timeout=300):
                with ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.reproduce_mols, smile_a, smile_b, mutation_rate)
                    try:
                        result = future.result(timeout=timeout)
                        return result
                    except TimeoutError:
                        print(f"Timeout occurred for smile: {smile_a}, {smile_b}")
                        return None
                    except Exception as e:
                        print(f"Error occurred for smile: {smile_a}, {smile_b}, error: {str(e)}")
                        return None

            #print('Expert Calculation initiated. Query: '+str(query_size))
            smiles = Parallel(n_jobs=pool.n_jobs, backend='loky')(
                delayed(reproduce_mols_wt_timeout)(smile_a, smile_b, self.mutation_rate)
                for smile_a, smile_b in zip(smiles_a, smiles_b)
            )
            #print('Expert Calculation completed.')
            #smiles = pool(
            #    delayed(self.reproduce_mols)(smile_a, smile_b, self.mutation_rate)
            #    for smile_a, smile_b in zip(smiles_a, smiles_b)
            #)

        smiles_list = list(filter(lambda smile: smile is not None, smiles))
        return smiles_list

    def reproduce_mols(
        self, parent_a: str, parent_b: str, mutation_rate: float, num_trials: int = 10
    ) -> List[str]:
    
        for _ in range(num_trials):
            try:
                parent_smiles = [parent_a, parent_b]
                child_smiles = None
                
                mol_a = Chem.MolFromSmiles(parent_a)
                mol_b = Chem.MolFromSmiles(parent_b)
                
                if mol_a is None or mol_b is None:
                    continue
                
                proba = random.random()
                if proba <= 0.8:
                    new_child = self.crossover_func(parent_a, parent_b)
                    if new_child is not None:
                        new_child = self.mutate_func(new_child, mutation_rate)
                else:
                    if random.random() <= 0.5:
                        new_child = self.mutate_func(parent_a, mutation_rate)
                    else:
                        new_child = self.mutate_func(parent_b, mutation_rate)
                
                if new_child is not None:
                    child_smiles = Chem.MolToSmiles(new_child)
                
                if child_smiles is not None and child_smiles not in parent_smiles:
                    break

            except:
                continue

        smiles = (
            Chem.MolToSmiles(new_child, isomericSmiles=True)
            if (new_child is not None) and (child_smiles is not None)
            else None
        )
        return smiles

    def reproduce_frags(self, smiles_list: List[str], mutation_rate: float) -> str:
        num_fragments = np.random.randint(2, 6)
        fragments = np.random.choice(smiles_list, num_fragments, replace=True).tolist()

        fragments_mol = [Chem.MolFromSmiles(frag) for frag in fragments]
        new_child = self.crossover_func(fragments_mol)  # type: ignore
        if new_child is not None:
            new_child = self.mutate_func(new_child, mutation_rate)

        smiles = (
            Chem.MolToSmiles(new_child, isomericSmiles=True)
            if new_child is not None
            else None
        )
        return smiles

