from pathlib import Path
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import hydra
from omegaconf import OmegaConf
from accelerate.state import PartialState

from accel_hydra.utils.config import register_omegaconf_resolvers
from accel_hydra.utils.lr_scheduler import (
    get_warmup_steps,
    get_dataloader_one_pass_outside_steps,
    get_total_training_steps,
    get_steps_inside_accelerator_from_outside_steps,
    get_dataloader_one_pass_steps_inside_accelerator,
    lr_scheduler_param_adapter,
)
from accel_hydra.models.common import CountParamsBase
from accel_hydra.trainer import Trainer
from accel_hydra.utils.data import init_dataloader_from_config

register_omegaconf_resolvers()

from transformers import SpeechT5ForSpeechToSpeech, SpeechT5Processor

def setup_resume_cfg(config):
    if "resume_from_checkpoint" in config["trainer"]:
        ckpt_dir = Path(config["trainer"]["resume_from_checkpoint"])
        exp_dir = ckpt_dir.parent.parent
        resumed_config = OmegaConf.load(exp_dir / "config.yaml")
        resumed_config["trainer"].update({
            "resume_from_checkpoint": ckpt_dir.__str__(),
            "logging_config": config["trainer"]
                              ["logging_config"],  # for resume wandb runs
        })
    elif config.get("auto_resume_from_latest_ckpt", False):
        exp_dir = Path(config["exp_dir"])
        ckpt_root = exp_dir / "checkpoints"
        if ckpt_root.is_dir() and any(p.is_dir() for p in ckpt_root.iterdir()):
            # use last ckpt
            ckpt_dir: Path = sorted((exp_dir / "checkpoints").iterdir())[-1]
            resumed_config = OmegaConf.load(exp_dir / "config.yaml")
            resumed_config["trainer"].update({
                "resume_from_checkpoint": ckpt_dir.__str__(),
                "logging_config": config["trainer"]
                                  ["logging_config"],  # for resume wandb runs
            })
        else:
            resumed_config = config
    else:
        resumed_config = config
    if "resume_from_checkpoint" in resumed_config["trainer"]:
        print(
            f'\n train will resume from checkpoint: {resumed_config["trainer"]["resume_from_checkpoint"]}\n '
        )
    else:
        print('\n train will start from scratch\n ')
    return resumed_config

from models.model import replace, freeze_all, unfreeze_adapters_by_name
def main():
    # processor = SpeechT5Processor.from_pretrained("/hpc_stor03/sjtu_home/yixuan.li/model_ckpt/speecht5_vc")
    model = SpeechT5ForSpeechToSpeech.from_pretrained("/hpc_stor03/sjtu_home/yixuan.li/model_ckpt/speecht5_vc").eval()
    replace(model, k=12, bottleneck_dim=128, dropout=0.1)
    freeze_all(model)
    unfreeze_adapters_by_name(model, name_keyword='adapter')

    configs = []

    @hydra.main(version_base=None, config_path="configs", config_name="train")
    def parse_config_from_command_line(config):
        config = OmegaConf.to_container(config, resolve=True)
        configs.append(config)

    parse_config_from_command_line()
    config = configs[0]

    if config.get("cfg_only", False):
        with open("./config.yaml", "w") as f:
            OmegaConf.save(config, f)
            print(f'config.yaml saved to {f.name}')
            return

    config = setup_resume_cfg(config)
    # helper state for accessing information about the current training environment
    state = PartialState()
    train_dataloader = init_dataloader_from_config(config["train_dataloader"])
    val_dataloader = init_dataloader_from_config(config["val_dataloader"])
    optimizer = hydra.utils.instantiate(
        config["optimizer"], params=model.parameters(), _convert_="all"
    )
    dataloader_one_pass_outside_steps = get_dataloader_one_pass_outside_steps(
        train_dataloader, state.num_processes
    )
    total_training_steps = get_total_training_steps(
        train_dataloader, config["epochs"], state.num_processes,
        config["epoch_length"]
    )
    dataloader_one_pass_steps_inside_accelerator = (
        get_dataloader_one_pass_steps_inside_accelerator(
            dataloader_one_pass_outside_steps,
            config["gradient_accumulation_steps"], state.num_processes
        )
    )
    num_training_updates = get_steps_inside_accelerator_from_outside_steps(
        total_training_steps, dataloader_one_pass_outside_steps,
        dataloader_one_pass_steps_inside_accelerator,
        config["gradient_accumulation_steps"], state.num_processes
    )

    num_warmup_steps = get_warmup_steps(
        **config["warmup_params"],
        dataloader_one_pass_outside_steps=dataloader_one_pass_outside_steps
    )

    num_warmup_updates = get_steps_inside_accelerator_from_outside_steps(
        num_warmup_steps, dataloader_one_pass_outside_steps,
        dataloader_one_pass_steps_inside_accelerator,
        config["gradient_accumulation_steps"], state.num_processes
    )

    lr_scheduler_config = lr_scheduler_param_adapter(
        config_dict=config["lr_scheduler"],
        num_training_steps=num_training_updates,
        num_warmup_steps=num_warmup_updates
    )

    lr_scheduler = hydra.utils.instantiate(
        lr_scheduler_config, optimizer=optimizer, _convert_="all"
    )

    loss_fn = hydra.utils.instantiate(config["loss_fn"], _convert_="all")
    trainer: Trainer = hydra.utils.instantiate(
        config["trainer"],
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_fn=loss_fn,
        _convert_="all"
    )
    trainer.config_dict = config  # assign here, don't instantiate it
    trainer.train(seed=config["seed"])


if __name__ == "__main__":
    main()
