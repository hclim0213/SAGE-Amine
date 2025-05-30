"""
Copyright (c) 2022 Hocheol Lim.
"""

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from sage.scoring.scoring_function import ScoringFunction
from joblib import Parallel
from rdkit import Chem
from torch_geometric.data import Batch

from sage.data import SmilesCharDictionary
from sage.logger.abstract_logger import AbstractLogger
from sage.memory import FragmentLibrary, MaxRewardPriorityMemory
from sage.models import (
    AbstractGeneratorHandler,
    ExplainerHandler,
    GeneticOperatorHandler,
)

from sage.models.apprentice import LSTMGenerator, TransformerGenerator, TransformerDecoderGenerator
from sage.models.apprentice import XTransformerGenerator
from sage.utils.featurizer import CanonicalFeaturizer
from sage.utils.sampling_handler import SamplingHandler
from sage.utils.smiles import calculate_similarity, canonicalize_and_score_smiles

from torch.utils.data import DataLoader, Dataset

class SMILESDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        return self.smiles_list[idx]

class Trainer:
    def __init__(
        self,
        apprentice_memory: MaxRewardPriorityMemory,
        expert_memory: MaxRewardPriorityMemory,
        apprentice_handler: AbstractGeneratorHandler,
        expert_handlers: List[GeneticOperatorHandler],
        char_dict: SmilesCharDictionary,
        num_keep: int,
        apprentice_sampling_batch_size: int,
        expert_sampling_batch_size: int,
        sampling_strategy: str,
        apprentice_training_batch_size: int,
        apprentice_training_steps: int,
        num_smiles_for_similarity: int,
        logger: AbstractLogger,
        fragment_library: Optional[FragmentLibrary] = None,
        init_smiles: Optional[List[str]] = None,
        explainer_handler: Optional[ExplainerHandler] = None,
        use_frozen_explainer: bool = False,
        evaluation_batch_size: Optional[int] = 1,
    ) -> None:
        self.apprentice_memory = apprentice_memory
        self.apprentice_mean_similarity = 1.0
        self.expert_memory = expert_memory

        self.apprentice_handler = apprentice_handler
        self.expert_handlers = expert_handlers
        self.explainer_handler = explainer_handler
        self.use_frozen_explainer = use_frozen_explainer

        self.char_dict = char_dict
        self.featurizer = CanonicalFeaturizer()

        self.num_keep = num_keep
        self.evaluation_batch_size = evaluation_batch_size
        self.apprentice_sampling_batch_size = apprentice_sampling_batch_size
        self.expert_sampling_batch_size = expert_sampling_batch_size
        self.apprentice_training_batch_size = apprentice_training_batch_size
        self.apprentice_training_steps = apprentice_training_steps

        num_experts = len(self.expert_handlers)
        self.num_experts = num_experts
        self.sampling_handler = SamplingHandler(
            num_experts, expert_sampling_batch_size, sampling_strategy
        )
        self.partial_query_sizes = (
            [  # Initialize the query sizes to uniform distribution at the beginning
                int(self.expert_sampling_batch_size / self.num_experts)
            ]
            * self.num_experts
        )

        self.logger = logger
        self.init_smiles = init_smiles
        self.num_smiles_for_similarity = num_smiles_for_similarity

        self.fragment_library = fragment_library

    def init(
        self, scoring_function: ScoringFunction, device: torch.device, pool: Parallel
    ) -> None:
        if len(self.init_smiles) > 0:  # type: ignore
            smiles, scores = canonicalize_and_score_smiles(
                smiles=self.init_smiles,  # type: ignore
                scoring_function=scoring_function,
                char_dict=self.char_dict,
                pool=pool,
                evaluation_batch_size=self.evaluation_batch_size,
            )
            self.apprentice_memory.add_list(smiles=smiles, scores=scores)
            self.expert_memory.add_list(smiles=smiles, scores=scores)

    def step(
        self, scoring_function: ScoringFunction, device: torch.device, pool: Parallel
    ) -> Tuple[List[str], List[float]]:
        # Generate and record SMILES from apprentice
        apprentice_smiles, apprentice_scores = self._update_memory_by_apprentice(
            scoring_function, device, pool
        )
        (
            best_apprentice_smiles,
            best_apprentice_scores,
            _,
        ) = self.apprentice_memory.get_elements()
        
        self.logger.log_text("Apprentice: ","Calcultion completed. SMILES: "+str(np.shape(apprentice_smiles))+", Scores: "+str(np.shape(apprentice_scores)))
        
        # Update fragment library if used
        if isinstance(self.fragment_library, FragmentLibrary):
            self.fragment_library.expand_fragment_lib(
                best_apprentice_smiles,
                best_apprentice_scores,
                device=device,
            )
            self.fragment_library.squeeze_by_max_length()
            self.logger.log_text("Fragment: ", "Calculation completed")

        # Generate and record SMILES from expert
        expert_smiles, expert_scores = self._update_memory_by_expert(
            scoring_function, device, pool
        )
        
        self.logger.log_text("Expert: ", "Calcultion completed. SMILES: "+str(np.shape(expert_smiles))+", Scores: "+str(np.shape(expert_scores)))
        
        loss, fit_size = self._train_apprentice_step(device)
        self.logger.log_metric("loss_apprentice", loss)
        self.logger.log_metric("fit_size", fit_size)

        smiles = apprentice_smiles + expert_smiles
        scores = apprentice_scores + expert_scores

        if isinstance(self.explainer_handler, ExplainerHandler):
            if not self.use_frozen_explainer:
                explainer_loss = self._train_explainer_step(smiles, scores, device)
                self.logger.log_metric("loss_explainer", explainer_loss)

        return smiles, scores
    
    def restart_step(
        self, restart_smiles: List[str], restart_scores: List[float], device: torch.device, pool: Parallel
    ) -> Tuple[List[str], List[float]]:
        
        # Add restart results to apprentice_memory
        self.apprentice_memory.add_list(smiles=restart_smiles, scores=restart_scores)
        self.apprentice_memory.squeeze_by_rank(top_k=self.num_keep)
        
        (
            best_apprentice_smiles,
            best_apprentice_scores,
            _,
        ) = self.apprentice_memory.get_elements()

        # Update fragment library if used
        if isinstance(self.fragment_library, FragmentLibrary):
            self.fragment_library.expand_fragment_lib(
                best_apprentice_smiles,
                best_apprentice_scores,
                device=device,
            )
            self.fragment_library.squeeze_by_max_length()
        
        # Add restart results to expert_memory
        self.expert_memory.add_list(smiles=restart_smiles, scores=restart_scores)
        self.expert_memory.squeeze_by_rank(top_k=self.num_keep)

        loss, fit_size = self._train_apprentice_step(device)
        self.logger.log_metric("loss_apprentice", loss)
        self.logger.log_metric("fit_size", fit_size)

        smiles = restart_smiles
        scores = restart_scores

        if isinstance(self.explainer_handler, ExplainerHandler):
            if not self.use_frozen_explainer:
                explainer_loss = self._train_explainer_step(smiles, scores, device)
                self.logger.log_metric("loss_explainer", explainer_loss)

        return smiles, scores
    
    def _update_memory_by_apprentice(
        self, scoring_function: ScoringFunction, device: torch.device, pool: Parallel
    ):
        with torch.no_grad():
            self.apprentice_handler.model.eval()
            
            try:
                context_smiles = self._get_all_smiles_from_memory()
            except:
                context_smiles = None

            smiles, _, _, _ = self.apprentice_handler.sample(
                num_samples=self.apprentice_sampling_batch_size,
                context_smiles=context_smiles,
                device=device,
            )

        canon_smiles, canon_scores = canonicalize_and_score_smiles(
            smiles=smiles,
            scoring_function=scoring_function,
            char_dict=self.char_dict,
            pool=pool,
            evaluation_batch_size=self.evaluation_batch_size,
        )

        self.apprentice_memory.add_list(smiles=canon_smiles, scores=canon_scores)
        self.apprentice_memory.squeeze_by_rank(top_k=self.num_keep)

        return canon_smiles, canon_scores

    def _update_memory_by_expert(
        self, scoring_function: ScoringFunction, device: torch.device, pool: Parallel
    ):
        self.logger.log_text("Expert: ", "Calculation Initiated #1")
        apprentice_smiles, _, _ = self.apprentice_memory.sample_batch(
            self.expert_sampling_batch_size
        )
        self.logger.log_text("Expert: ", "Calculation Initiated #2")
        
        canon_smiles: List[str] = []
        canon_scores: List[float] = []

        for expert_idx in range(self.num_experts):
            expert = self.expert_handlers[expert_idx]
            if expert.crossover_func.__name__ == "fragment_crossover":
                mating_pool, _ = self.fragment_library.get_fragments()  # type: ignore
            else:
                mating_pool = apprentice_smiles
            
            self.logger.log_text("Expert: ", "Calculation Initiated #3")
            query_smiles = expert.query(
                query_size=self.partial_query_sizes[expert_idx],
                apprentice_mean_similarity=self.apprentice_mean_similarity,
                mating_pool=mating_pool,
                pool=pool,
            )
            self.logger.log_text("Expert: ", "Calculation Initiated #4")
            partial_smiles, partial_scores = canonicalize_and_score_smiles(
                smiles=query_smiles,
                scoring_function=scoring_function,
                char_dict=self.char_dict,
                pool=pool,
                evaluation_batch_size=self.evaluation_batch_size,
            )
            self.logger.log_text("Expert: ", "Calculation Initiated #5")
            self.expert_memory.add_list(
                smiles=partial_smiles, scores=partial_scores, expert_id=expert_idx
            )
            canon_smiles += partial_smiles
            canon_scores += partial_scores

        self.logger.log_metric("mutation_rate", self.expert_handlers[0].mutation_rate)
        self.expert_memory.squeeze_by_rank(top_k=self.num_keep)

        expert_ratios = [
            query_size / self.expert_sampling_batch_size
            for query_size in self.partial_query_sizes
        ]
        self.logger.log_values("expert_ratios", expert_ratios)

        # Update the partial query sizes for the next round
        _, _, expert_ids = self.expert_memory.get_elements()
        self.partial_query_sizes = self.sampling_handler.calculate_partial_query_size(
            expert_ids
        )

        return canon_smiles, canon_scores

    def _train_apprentice_step(self, device: torch.device) -> Tuple[float, int]:
        average_loss = 0.0

        all_smiles = self._get_all_smiles_from_memory()
        self.apprentice_handler.model.train()  # type: ignore
        
        dataset = SMILESDataset(all_smiles)
        dataloader = DataLoader(dataset, batch_size=self.apprentice_training_batch_size, shuffle=True)

        for _ in range(self.apprentice_training_steps):
            for smiles_batch in dataloader:
                loss = self.apprentice_handler.train_on_batch(smiles=smiles_batch, device=device)
                average_loss += loss / len(dataloader)
        
        #for _ in range(self.apprentice_training_steps):
        #    for i in range(0, len(all_smiles), self.apprentice_training_batch_size):
        #        smiles_batch = all_smiles[i:i + self.apprentice_training_batch_size]
        #        loss = self.apprentice_handler.train_on_batch(smiles=smiles_batch, device=device)
        #        average_loss += loss / (len(all_smiles) // self.apprentice_training_batch_size)
        
        #for _ in range(self.apprentice_training_steps):
        #    smiles = random.choices(all_smiles, k=self.apprentice_training_batch_size)
        #    loss = self.apprentice_handler.train_on_batch(smiles=smiles, device=device)
        #    average_loss += loss / self.apprentice_training_steps

        fit_size = len(all_smiles)

        return average_loss, fit_size

    def _train_explainer_step(
        self, all_smiles: List[str], all_scores: List[float], device: torch.device
    ) -> float:
        average_loss = 0.0
        self.explainer_handler.model.train()  # type: ignore

        for _ in range(self.apprentice_training_steps):
            idxs = random.choices(
                range(len(all_smiles)), k=self.apprentice_training_batch_size
            )
            smiles = [all_smiles[idx] for idx in idxs]
            scores = [all_scores[idx] for idx in idxs]

            data_list = [
                self.featurizer.process(Chem.MolFromSmiles(smile), score)
                for smile, score in zip(smiles, scores)
            ]
            batch_data = Batch.from_data_list(data_list)
            loss = self.explainer_handler.train_on_graph_batch(  # type: ignore
                batch_data, device=device
            )
            average_loss += loss / self.apprentice_training_steps

        return average_loss

    def _get_all_smiles_from_memory(self) -> List[str]:
        apprentice_smiles, _, _ = self.apprentice_memory.get_elements()
        expert_smiles, _, _ = self.expert_memory.get_elements()
        all_smiles = list(set(apprentice_smiles + expert_smiles))
        return all_smiles

    def log_fragments(self) -> None:
        if self.fragment_library is not None:
            fragments, scores = self.fragment_library.get_fragments()
            for fragment, score in zip(fragments, scores):
                self.logger.log_text("fragment_smile", fragment)
                self.logger.log_metric("fragment_score", score)
