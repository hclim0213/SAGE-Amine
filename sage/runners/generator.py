"""
Copyright (c) 2022 Hocheol Lim.
"""
import os
from typing import List, Optional, Union

import torch
from sage.scoring.goal_directed_generator import GoalDirectedGenerator
from sage.scoring.scoring_function import ScoringFunction
from joblib import Parallel
from tqdm import tqdm
import numpy as np

from sage.memory import Recorder
from sage.runners.trainer import Trainer

class Generator(GoalDirectedGenerator):
    def __init__(
        self,
        trainer: Trainer,
        recorder: Recorder,
        num_steps: int,
        device: torch.device,
        scoring_num_list: List[int],
        num_jobs: int,
        dataset_type: Optional[str] = None,
        early_stop: Optional[bool] = False,
        restart: Optional[bool] = False,
        restart_dir: Optional[str] = "",
    ) -> None:
        self.trainer = trainer
        self.recorder = recorder
        self.num_steps = num_steps
        self.device = device
        self.scoring_num_list = scoring_num_list
        self.dataset_type = dataset_type
        self.early_stop = early_stop

        self.pool = Parallel(n_jobs=num_jobs)
        self.num_jobs = num_jobs
        self.restart = restart
        self.restart_dir = restart_dir
        
    def generate_optimized_molecules(
        self,
        scoring_function: ScoringFunction,
        number_molecules: int,
        starting_population: Optional[List[str]] = None,
    ) -> List[str]:
        self.trainer.init(
            scoring_function=scoring_function, device=self.device, pool=self.pool,
        )
        
        for step in tqdm(range(self.num_steps)):
            
            if self.restart == True and self.restart_dir != "":
                try:
                    with open(self.restart_dir+'/step_'+str(step+1).zfill(3)+'_smiles.txt', 'r') as file:
                        restart_smiles = [line.strip() for line in file.readlines()]
                    
                    with open(self.restart_dir+'/step_'+str(step+1).zfill(3)+'_scores.txt', 'r') as file:
                        restart_scores = [float(line.strip()) for line in file.readlines()]
                    
                    smiles, scores = self.trainer.restart_step(
                        restart_smiles, restart_scores, device=self.device, pool=self.pool
                    )
                    
                    self.trainer.logger.log_text('Step '+str(step+1), "Restart succeeded. SMILES: "+str(np.shape(smiles))+", Scores: "+str(np.shape(scores)))
                    
                except Exception as e:
                    self.trainer.logger.log_text('Step '+str(step+1), "Restart failed due to error "+str(e))
                    self.restart = False
                    self.trainer.logger.log_text('Step '+str(step+1), "Restart completed and calculation initiated")
            else:
                self.restart = False
            
            if self.restart is False:
                smiles, scores = self.trainer.step(
                    scoring_function=scoring_function, device=self.device, pool=self.pool
                )
                self.trainer.logger.log_text('Step '+str(step+1), "Calculation completed. SMILES: "+str(np.shape(smiles))+", Scores: "+str(np.shape(scores)))

            self.recorder.add_list(smiles=smiles, scores=scores)
            current_score = self.recorder.get_and_log_score()

            temp_file_smiles = open(self.recorder.save_dir+'step_'+str(step+1).zfill(3)+'_smiles.txt', 'w')
            temp_file_smiles.write('\n'.join(smiles))

            temp_file_scores = open(self.recorder.save_dir+'step_'+str(step+1).zfill(3)+'_scores.txt', 'w')
            temp_file_scores.write('\n'.join(map(str, scores)))

            temp_file_smiles.close()
            temp_file_scores.close()
            
            if self.dataset_type == "guacamol" or self.early_stop == True:
                if current_score == 1.0:
                    break
                if round(current_score,3) == 1.0:
                    break

        self.recorder.log_final()
        self.trainer.log_fragments()
        best_smiles, best_scores = self.recorder.get_topk(top_k=number_molecules)
        return best_smiles

