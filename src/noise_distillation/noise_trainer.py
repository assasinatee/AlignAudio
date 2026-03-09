from dataclasses import dataclass
import gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils._pytree import tree_map
from omegaconf import OmegaConf

from accel_hydra.trainer import Trainer
from accel_hydra.utils.logging import LoggingLogger

from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech
from models.model import replace

@dataclass(kw_only=True)
class NoiseTrainer(Trainer):
    logging_file: str | Path
    model_weight_path: str | Path
    device: str

    def on_train_start(self):
        super().on_train_start()
        if self.accelerator.is_main_process:
            self.logger = LoggingLogger(self.logging_file).create_instance()
        self.processor = SpeechT5Processor.from_pretrained(self.model_weight_path)

    def training_step(self, batch, batch_idx):
        clean_wav = batch["clean_wav"]      
        noisy_wav = batch["noisy_wav"]     
        feat      = batch["encoder_feat"]  
        # print("输入",clean_wav[0].shape, clean_wav[0].shape, feat[0].shape)

        clean_inputs = self.processor(
            audio=clean_wav,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        noisy_inputs = self.processor(
            audio=noisy_wav,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        # move to device
        clean_inputs = {k: v.to(self.device) for k, v in clean_inputs.items()}
        noisy_inputs = {k: v.to(self.device) for k, v in noisy_inputs.items()}
        
        # forward for encoder features
        clean_enc = self.model.speecht5.encoder(**clean_inputs).last_hidden_state  # (B, T_max, D)
        noisy_enc = self.model.speecht5.encoder(**noisy_inputs).last_hidden_state  # (B, T_max, D)

        loss = 0.0
        B = len(feat)
        
        print("截断前",clean_enc[0].shape, noisy_enc[0].shape, feat[0].shape)
        for j in range(B):
            clean_feat_j = clean_enc[j]
            noisy_feat_j = noisy_enc[j]
            target_feat_j = feat[j].to(self.device)  

            assert clean_feat_j.shape[0] >= target_feat_j.shape[0], \
                f"clean frame mismatch: {clean_feat_j.shape[0]} vs {target_feat_j.shape[0]}"
            clean_feat_j = clean_feat_j[:target_feat_j.shape[0],]
            assert noisy_feat_j.shape[0] >= target_feat_j.shape[0], \
                f"noisy frame mismatch: {noisy_feat_j.shape[0]} vs {target_feat_j.shape[0]}"
            noisy_feat_j = noisy_feat_j[:target_feat_j.shape[0],]

            loss += 0.3 * self.loss_fn(clean_feat_j, target_feat_j) + \
                    0.7 * self.loss_fn(noisy_feat_j, target_feat_j)

        loss = loss / B

        lr = self.optimizer.param_groups[0]["lr"]
        self.accelerator.log({"train/lr": lr}, step=self.step)

        # print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        return loss

    def on_validation_start(self):
        # 初始化验证统计
        self.validation_stats = {"loss_sum": 0.0, "num_batches": 0}

    def validation_step(self, batch, batch_idx):
        clean_wav = batch["clean_wav"]      
        noisy_wav = batch["noisy_wav"]     
        feat      = batch["encoder_feat"]  
        # print("输入",clean_wav[0].shape, clean_wav[0].shape, feat[0].shape)
        with torch.no_grad():
            clean_inputs = self.processor(
                audio=clean_wav,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            noisy_inputs = self.processor(
                audio=noisy_wav,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            # move to device
            clean_inputs = {k: v.to(self.device) for k, v in clean_inputs.items()}
            noisy_inputs = {k: v.to(self.device) for k, v in noisy_inputs.items()}
            
            # forward for encoder features
            clean_enc = self.model.speecht5.encoder(**clean_inputs).last_hidden_state  # (B, T_max, D)
            noisy_enc = self.model.speecht5.encoder(**noisy_inputs).last_hidden_state  # (B, T_max, D)

            loss = 0.0
            B = len(feat)
            
            print("截断前",clean_enc[0].shape, noisy_enc[0].shape, feat[0].shape)
            for j in range(B):
                clean_feat_j = clean_enc[j]
                noisy_feat_j = noisy_enc[j]
                target_feat_j = feat[j].to(self.device)  

                assert clean_feat_j.shape[0] >= target_feat_j.shape[0], \
                    f"clean frame mismatch: {clean_feat_j.shape[0]} vs {target_feat_j.shape[0]}"
                clean_feat_j = clean_feat_j[:target_feat_j.shape[0],]
                assert noisy_feat_j.shape[0] >= target_feat_j.shape[0], \
                    f"noisy frame mismatch: {noisy_feat_j.shape[0]} vs {target_feat_j.shape[0]}"
                noisy_feat_j = noisy_feat_j[:target_feat_j.shape[0],]

                loss += 0.3 * self.loss_fn(clean_feat_j, target_feat_j) + \
                        0.7 * self.loss_fn(noisy_feat_j, target_feat_j)

            loss = loss / B

            lr = self.optimizer.param_groups[0]["lr"]
            self.accelerator.log({"train/lr": lr}, step=self.step)

            # 累加
            self.validation_stats["loss_sum"] += loss
            self.validation_stats["num_batches"] += len(clean_wav)

    def get_val_metrics(self):
        avg_loss = self.validation_stats["loss_sum"] / max(1, self.validation_stats["num_batches"])
        return {"loss": avg_loss}

    def on_validation_end(self):
        avg_loss = self.validation_stats["loss_sum"] / max(1, self.validation_stats["num_batches"])
        self.accelerator.log({"loss": avg_loss}, step=self.step)

    def on_train_epoch_end(self):
        if self.accelerator.is_main_process:
            self.logger.info(f"training epoch {self.epoch} ended")


@dataclass(kw_only=True)
class MnistTrainer(Trainer):

    logging_file: str | Path

    def on_train_start(self):
        super().on_train_start()
        if self.accelerator.is_main_process:
            self.logger = LoggingLogger(self.logging_file).create_instance()
        

    def training_step(self, batch, batch_idx):
        features, labels = batch
        preds = self.model(features)
        loss = self.loss_fn(preds, labels)
        lr = self.optimizer.param_groups[0]["lr"]
        self.accelerator.log({"train/lr": lr}, step=self.step)
        return loss

    def on_validation_start(self):
        self.validation_stats = {"accurate": 0, "num_elems": 0}

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        preds = self.model(features)
        predictions = preds.argmax(dim=-1)
        output = {"predictions": predictions, "labels": labels}
        output = self.accelerator.gather_for_metrics(output)
        accurate_preds = (output["predictions"] == output["labels"])
        self.validation_stats["accurate"] += accurate_preds.long().sum()
        self.validation_stats["num_elems"] += accurate_preds.shape[0]

    def get_val_metrics(self):
        return {
            "accuracy":
                self.validation_stats["accurate"].item() /
                self.validation_stats["num_elems"]
        }

    def on_validation_end(self):
        eval_metric = self.validation_stats["accurate"].item(
        ) / self.validation_stats["num_elems"]
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.accelerator.print(
            f"epoch[{self.epoch}]@{nowtime} --> eval_metric= {100 * eval_metric:.2f}%"
        )
        if self.accelerator.is_main_process:
            self.logger.info(
                f"epoch[{self.epoch}]@{nowtime} --> eval_metric= {100 * eval_metric:.2f}%"
            )
        self.accelerator.log({"val/accuracy": eval_metric}, step=self.step)

    def on_train_epoch_end(self):
        if self.accelerator.is_main_process:
            self.logger.info(f"training epoch {self.epoch} ended")