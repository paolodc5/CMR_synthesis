import argparse
import os
import sys
import json
import numpy as np
from dataclasses import asdict
from sklearn.model_selection import train_test_split
import argparse
import logging

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm import tqdm

from datasets import FastDatasetDIDC
from utils import setup_logger, set_reproducibility
from gan_basic import DiscriminatorModel
from unet_advanced import UNetAdvanced as GeneratorModel
from mt_DIDC_config import GROUPING_RULES, NEW_LABELS

from tx_trainer import GANTrainer, UnetTrainer
from tx_config import GANTrainerConfig, UnetTrainerConfig
from tx_bssfps_simulator import bSSFPSimulator

# Setup logger
logger = get_logger(__name__, log_level="INFO")


class CustomDatasetTexturizer(FastDatasetDIDC):
    def __init__(self, data_path: str, logger: logging.Logger, file_list: list[str]=None, global_scale: float = 1.0):
        super().__init__(data_path=data_path, file_list=file_list)
        self.global_scale = global_scale

        pd_max = 200.0
        t1_max = 2000.0
        t2_max = 500.0

        config_path = os.path.join(self.data_path, 'config_properties.json')
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                pd_max = config_data['PD_max']
                t1_max = config_data['T1_max']
                t2_max = config_data['T2_max']
                logger.info(f"Loaded property max values from config: PD_max={pd_max}, T1_max={t1_max}, T2_max={t2_max}")

                if config_data['save_normalized']:
                    logger.info("Properties are saved normalized to [0,1]. Using max values for scaling)")
                    self.props_scale = torch.tensor([pd_max, t1_max, t2_max], dtype=torch.float32).view(3, 1, 1)
                else:
                    logger.info("Properties are saved in physical units. No scaling applied.")
                    self.props_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).view(3, 1, 1)

        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}. I'm normalizing using default property max values: PD_max={pd_max}, T1_max={t1_max}, T2_max={t2_max}")

        except KeyError as e:
            logger.warning(f"Missing key in config file: {e}. I'm normalizing using default property max values: PD_max={pd_max}, T1_max={t1_max}, T2_max={t2_max}")



    def __getitem__(self, idx):
        # load the original sample and take only the mask
        original_sample = super().__getitem__(idx)
        label = original_sample['multiClassMask']

        # load mri slice from the same pat/slice
        pat_id, slice_idx = self.samples[idx]
        pat_path = os.path.join(self.data_path, f"{pat_id}_img.npy")
        props_path = os.path.join(self.data_path, f"{pat_id}_props.npy")

        mri_slice = np.load(pat_path, mmap_mode='r')[slice_idx]
        try: 
            props_slice = np.load(props_path, mmap_mode='r')[slice_idx] # if slices are the first dimension
        except IndexError:
            props_slice = np.load(props_path, mmap_mode='r')[..., slice_idx]

        mri_slice_tensor = torch.from_numpy(mri_slice.copy() / self.global_scale).float()
        props_slice_tensor = torch.from_numpy(props_slice.copy()).float() * self.props_scale # crucial because properties are saved normalized to [0,1], so we need to scale them back to their actual range

        return {'input_label': label, 'mri_slice': mri_slice_tensor, 'props_slice': props_slice_tensor}


def compute_mean_99th_perc_scale(dataset, max_samples=5000):
    percentiles = []
    
    indices = np.random.choice(len(dataset), min(len(dataset), max_samples), replace=False)

    for idx in tqdm(indices, desc="Compute 99th perc. slices"):
        mri_slice = dataset[idx]['mri_slice'].numpy()
        no_bkg_pixels = mri_slice[mri_slice > 1e-3]

        if len(no_bkg_pixels) > 0:
            perc_99 = np.percentile(no_bkg_pixels, 99)
            percentiles.append(perc_99)

    return float(np.mean(percentiles))


def main():

    parser = argparse.ArgumentParser(description="Training Pipeline per UNet e GAN")
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        choices=['unet', 'gan'], 
        help="Choose which model to train: 'unet' or 'gan'"
    )
    args = parser.parse_args()

    if args.mode == 'gan':
        config = GANTrainerConfig()
    else:
        config = UnetTrainerConfig()

    set_reproducibility(config.seed)
    
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=config.run_dir
    )

    if accelerator.is_main_process:
        os.makedirs(config.run_dir, exist_ok=True)
        setup_logger(config.run_dir)
        accelerator.init_trackers("tb_tracker")

    logger.info(f"Initialized {args.mode} training")

    dataset_config_path = os.path.join(config.data_path, "dataset_config.json")
    config_properties_path = os.path.join(config.data_path, "config_properties.json")
    if os.path.exists(dataset_config_path) and os.path.exists(config_properties_path):
        with open(dataset_config_path, "r") as f:
            dataset_config = json.load(f)
        with open(config_properties_path) as f:
            properties_config = json.load(f)
    else:
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path} or {config_properties_path}")

    all_files = sorted(list(set([f.replace('_props.npy', '') for f in os.listdir(config.data_path) if f.endswith('props.npy')])))
    train_files, val_files = train_test_split(all_files, test_size=config.val_fraction, random_state=config.seed)

    train_dataset = CustomDatasetTexturizer(data_path=config.data_path, file_list=train_files, logger=logger)
    val_dataset = CustomDatasetTexturizer(data_path=config.data_path, file_list=val_files, logger=logger)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size_per_gpu, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size_per_gpu, shuffle=False, num_workers=config.num_workers)

    logger.info(f"Computing mri slices scale factor (99th percentile) on {min(len(train_dataset), config.max_sample_statistics)} slices")
    global_scale = compute_mean_99th_perc_scale(train_dataset, max_samples=config.max_sample_statistics)
    train_dataset.global_scale = global_scale
    val_dataset.global_scale = global_scale
    config.global_scale = global_scale
    logger.info(f"Computed global scale factor for MRI slices: {global_scale:.4f}")

    if accelerator.is_main_process:
        with open(f"{config.run_dir}/train_val_split.json", "w") as f:
            json.dump({'train_indices': train_files, 'val_indices': val_files}, f, indent=4)
        with open(f"{config.run_dir}/grouping_rules_and_labels.json", "w") as f:
            json.dump({'grouping_rules': GROUPING_RULES, 'new_labels': NEW_LABELS}, f, indent=4)
        with open(f"{config.run_dir}/training_config.json", "w") as f:
            json.dump({**asdict(config), **dataset_config, **properties_config}, f, indent=4)

    bssfp_sim = bSSFPSimulator(config.bssfp_model)

    if args.mode == 'gan':
        gen = GeneratorModel(**asdict(config.gen_model)) # input channels: 1 or 22
        discr = DiscriminatorModel(**asdict(config.discr_model)) # 4 channels: (1 condition + 3 offset)

        opt_G = torch.optim.AdamW(gen.parameters(), lr=config.lr_gen)
        opt_D = torch.optim.AdamW(discr.parameters(), lr=config.lr_discr)
 
        gen, discr, opt_G, opt_D, train_loader, val_dataloader, bssfp_sim = accelerator.prepare(
            gen, discr, opt_G, opt_D, train_loader, val_dataloader, bssfp_sim
        )
        
        trainer = GANTrainer(
            config=config,
            gen=gen,
            discr=discr,
            bssfp_sim=bssfp_sim,
            opt_G=opt_G,
            opt_D=opt_D,
            train_loader=train_loader,
            val_loader=val_dataloader,
            accelerator=accelerator,
            logger=logger
        )
    
    if args.mode == 'unet':
        model = GeneratorModel(**asdict(config.gen_model)) # input channels: 1 or 22
        opt_G = torch.optim.Adam(model.parameters(), lr=config.lr)

        model, opt_G, train_loader, val_dataloader, bssfp_sim = accelerator.prepare(
            model, opt_G, train_loader, val_dataloader, bssfp_sim
        )

        trainer = UnetTrainer(
            config=config,
            model=model,
            bssfp_sim=bssfp_sim,
            opt=opt_G,
            train_loader=train_loader,
            val_loader=val_dataloader,
            accelerator=accelerator,
            logger=logger
        )
        
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Training crashed with the following error:")
        sys.exit(1)
    
    

