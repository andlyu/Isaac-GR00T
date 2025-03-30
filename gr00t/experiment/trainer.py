# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
import transformers
from torch.utils.data import Dataset, Sampler
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)
import torch.nn as nn
import torch

from typing import Dict, List, Optional, Tuple, Union, Any
import time

class BaseSampler(Sampler):
    """Sampler for dataset, which enables `set_epoch` for Dataset.
    `set_epoch` will be called by huggingface Trainer at the end of each epoch.
    `shuffle` is also supported for training set shuffling
    """

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # must not add rank here, or randomization will be different for each rank
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            # this is important for dataset
            self.data_source.set_epoch(epoch)

    def __len__(self):
        return len(self.data_source)


class DualBrainTrainer(transformers.Trainer):
    def __init__(self, **kwargs):
        self.compute_dtype = kwargs.pop("compute_dtype")
        super().__init__( **kwargs)

    def _get_train_sampler(self):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):    
        
        from torch.utils.data import Subset
        total_length = len(eval_dataset)
        indices = list(range(0, total_length, 20))
        eval_subset = Subset(eval_dataset, indices)
        return BaseSampler(eval_subset, shuffle=False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def prediction_step(
        self,
        model: nn.Module,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on the model using inputs.
        """
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            outputs = model(inputs)
            
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss'].mean().detach()
            else:
                loss = None
                
            return (loss, None, None)
        
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        # Use the existing evaluation dataset if none is provided
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        # Get the evaluation dataloader
        eval_dataloader = DataLoader(
            eval_dataset if eval_dataset is not None else self.eval_dataset,
            sampler=self._get_eval_sampler(eval_dataset if eval_dataset is not None else self.eval_dataset),
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=0,  # Use 0 workers to avoid multiprocessing issues
            pin_memory=self.args.dataloader_pin_memory,
        )        
        # Initialize metrics
        metrics = {}
        
        # Start timing
        start_time = time.time()
        
        # Collect losses manually from the model
        losses = []
        model = self.model
        model.eval()
        num_samples = 0
        num_steps = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                try:
                    batch = self._prepare_inputs(batch)
                    outputs = model(batch)
                    
                    # Check if outputs is a BatchFeature object with a 'loss' attribute
                    if hasattr(outputs, 'loss') or (isinstance(outputs, dict) and 'loss' in outputs):
                        # Handle both BatchFeature and dictionary cases
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                        losses.append(loss.detach().cpu().item())
                    
                    num_samples += len(next(iter(batch.values())))  # Estimate batch size
                    num_steps += 1
                except Exception as e:
                    print(f"Error processing batch {i}: {e}")
                    continue  # Skip problematic batches
        
        # Calculate average loss if we collected any losses
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        # Calculate runtime metrics
        runtime = time.time() - start_time
        metrics[f"{metric_key_prefix}_loss"] = avg_loss
        metrics[f"{metric_key_prefix}_runtime"] = runtime
        metrics[f"{metric_key_prefix}_samples_per_second"] = num_samples / runtime
        metrics[f"{metric_key_prefix}_steps_per_second"] = num_steps / runtime
        metrics["epoch"] = self.state.epoch
        
        # Log and return metrics
        self.log(metrics)
        return metrics

    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            state_dict = self.model.state_dict()

        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict)

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
