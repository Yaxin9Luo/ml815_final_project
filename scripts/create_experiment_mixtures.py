"""
Create 3 mixed training datasets from per-domain tokenized bins,
and generate corresponding training configs.

Each mixture uses a 3B token budget with different domain proportions.
Also generates training config files in config/generated/.

Usage:
    python scripts/create_experiment_mixtures.py
"""

import os
import numpy as np
from pathlib import Path
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Paths ---
TOKENIZED_DATA_DIR = '/data/yaxin/pretrain_data_analysis/data/slimpajama_tokenized'
OUTPUT_BASE_DIR = '/data/yaxin/pretrain_data_analysis/data'
CONFIG_BASE_FILE = '/data/yaxin/pretrain_data_analysis/config/train_gpt2_medium_slimpajama.py'
CONFIG_OUTPUT_DIR = '/data/yaxin/pretrain_data_analysis/config/generated'

# --- Budget ---
TRAIN_BUDGET_TOKENS = 3_000_000_000
VAL_TOKENS_PER_CATEGORY = 5_000_000

# --- Three experiment mixtures ---
EXPERIMENTS = [
    {
        'name': 'balanced',
        'proportions': {
            'RedPajamaCommonCrawl':   0.20,
            'RedPajamaC4':            0.20,
            'RedPajamaBook':          0.20,
            'RedPajamaWikipedia':     0.10,
            'RedPajamaArXiv':         0.10,
            'RedPajamaGithub':        0.10,
            'RedPajamaStackExchange': 0.10,
        }
    },
    {
        'name': 'book_heavy',
        'proportions': {
            'RedPajamaCommonCrawl':   0.05,
            'RedPajamaC4':            0.05,
            'RedPajamaBook':          0.70,
            'RedPajamaWikipedia':     0.10,
            'RedPajamaArXiv':         0.05,
            'RedPajamaGithub':        0.03,
            'RedPajamaStackExchange': 0.02,
        }
    },
    {
        'name': 'web_heavy',
        'proportions': {
            'RedPajamaCommonCrawl':   0.45,
            'RedPajamaC4':            0.30,
            'RedPajamaBook':          0.02,
            'RedPajamaWikipedia':     0.07,
            'RedPajamaArXiv':         0.03,
            'RedPajamaGithub':        0.05,
            'RedPajamaStackExchange': 0.08,
        }
    },
]


def create_mixed_dataset(tokenized_data_path, output_path, proportions,
                         total_token_budget, val_tokens_per_category):
    """
    Creates a mixed training and validation dataset from pre-tokenized binary files.
    """
    base_path = Path(tokenized_data_path)
    output_base_path = Path(output_path)
    output_base_path.mkdir(parents=True, exist_ok=True)

    # Validate proportions
    prop_sum = sum(proportions.values())
    if abs(prop_sum - 1.0) > 1e-6:
        raise ValueError(f"Proportions must sum to 1.0, got {prop_sum}")

    # --- Create Validation Set ---
    logging.info("Creating validation set...")
    val_data = []
    for category in proportions.keys():
        bin_file = base_path / category / "train.bin"
        if not bin_file.exists():
            logging.warning(f"  Validation: {bin_file} not found. Skipping.")
            continue

        m = np.memmap(bin_file, dtype=np.uint16, mode='r')
        actual_val_tokens = min(val_tokens_per_category, len(m) // 10)
        val_data.append(np.array(m[-actual_val_tokens:]))
        logging.info(f"  - {category}: took {actual_val_tokens:,} tokens from end for validation")

    val_arr = np.concatenate(val_data).astype(np.uint16)
    val_filename = output_base_path / 'val.bin'
    logging.info(f"Writing validation data to {val_filename} ({len(val_arr):,} tokens)")
    val_arr.tofile(val_filename)

    # --- Create Training Set ---
    logging.info("Creating training set...")
    train_filename = output_base_path / 'train.bin'

    train_chunks = []
    total_tokens_written = 0

    for category, proportion in proportions.items():
        num_tokens_to_take = int(total_token_budget * proportion)

        bin_file = base_path / category / "train.bin"
        if not bin_file.exists():
            logging.warning(f"  Training: {bin_file} not found. Skipping.")
            continue

        m = np.memmap(bin_file, dtype=np.uint16, mode='r')
        # Exclude validation tokens from the end
        available = len(m) - val_tokens_per_category
        if available <= 0:
            logging.warning(f"  {category}: not enough tokens after validation split")
            continue

        actual_take = min(num_tokens_to_take, available)
        if actual_take < num_tokens_to_take:
            logging.warning(f"  {category}: only {available:,} tokens available, "
                          f"requested {num_tokens_to_take:,}")

        chunk = np.array(m[:actual_take])
        train_chunks.append(chunk)
        total_tokens_written += len(chunk)
        logging.info(f"  - {category}: {len(chunk):,} tokens ({proportion*100:.0f}%)")

    # Concatenate and write
    logging.info(f"Concatenating {len(train_chunks)} domain chunks...")
    train_arr = np.concatenate(train_chunks).astype(np.uint16)

    # Shuffle at block level to interleave domains
    # (shuffle 1024-token blocks so domains are mixed during training)
    block_size = 1024
    n_complete_blocks = len(train_arr) // block_size
    train_arr_trimmed = train_arr[:n_complete_blocks * block_size]
    blocks = train_arr_trimmed.reshape(n_complete_blocks, block_size)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(blocks)
    train_arr_shuffled = blocks.reshape(-1)

    logging.info(f"Writing training data to {train_filename} ({len(train_arr_shuffled):,} tokens)")
    train_arr_shuffled.tofile(train_filename)

    logging.info(f"Total training tokens: {len(train_arr_shuffled):,}")
    logging.info(f"Total validation tokens: {len(val_arr):,}")
    return train_filename, val_filename


def generate_config(exp_name, config_base_file, config_output_dir, dataset_name):
    """Generate experiment-specific training config."""
    config_dir = Path(config_output_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    with open(config_base_file, 'r') as f:
        base_content = f.read()

    override = f"""
# --- Overrides for experiment: {exp_name} ---
out_dir = 'out/gpt2_medium_{exp_name}'
dataset = '{dataset_name}'
wandb_run_name = '{exp_name}'
"""

    config_file = config_dir / f"config_{exp_name}.py"
    with open(config_file, 'w') as f:
        f.write(base_content + override)

    logging.info(f"Generated config: {config_file}")
    return config_file


def main():
    logging.info("=== Creating Experiment Mixtures ===")

    for exp in EXPERIMENTS:
        exp_name = exp['name']
        proportions = exp['proportions']

        logging.info(f"\n{'='*60}")
        logging.info(f"Experiment: {exp_name}")
        logging.info(f"{'='*60}")
        logging.info(f"Proportions: {proportions}")

        # Create mixed dataset
        exp_data_dir = Path(OUTPUT_BASE_DIR) / f"slimpajama-3b-{exp_name}"
        create_mixed_dataset(
            tokenized_data_path=TOKENIZED_DATA_DIR,
            output_path=str(exp_data_dir),
            proportions=proportions,
            total_token_budget=TRAIN_BUDGET_TOKENS,
            val_tokens_per_category=VAL_TOKENS_PER_CATEGORY,
        )

        # Generate training config
        dataset_name = f"slimpajama-3b-{exp_name}"
        generate_config(exp_name, CONFIG_BASE_FILE, CONFIG_OUTPUT_DIR, dataset_name)

    # Print summary
    logging.info(f"\n{'='*60}")
    logging.info("ALL EXPERIMENTS CREATED")
    logging.info(f"{'='*60}")
    for exp in EXPERIMENTS:
        name = exp['name']
        data_dir = Path(OUTPUT_BASE_DIR) / f"slimpajama-3b-{name}"
        train_file = data_dir / 'train.bin'
        val_file = data_dir / 'val.bin'
        train_size = train_file.stat().st_size if train_file.exists() else 0
        val_size = val_file.stat().st_size if val_file.exists() else 0
        logging.info(f"  {name}:")
        logging.info(f"    train.bin: {train_size / 1e9:.2f} GB ({train_size // 2:,} tokens)")
        logging.info(f"    val.bin:   {val_size / 1e6:.1f} MB ({val_size // 2:,} tokens)")
        logging.info(f"    config:    config/generated/config_{name}.py")

    logging.info("\nTo train, run:")
    for exp in EXPERIMENTS:
        name = exp['name']
        logging.info(f"  deepspeed --include=localhost:4,5,6,7 train_deepspeed.py config/generated/config_{name}.py")


if __name__ == '__main__':
    main()
