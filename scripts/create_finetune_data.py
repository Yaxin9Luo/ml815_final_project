import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_mixed_dataset(tokenized_data_path, output_path, proportions, total_token_budget, val_tokens_per_category=5_000_000):
    """
    Creates a mixed training and validation dataset from pre-tokenized binary files.

    Args:
        tokenized_data_path (str): Path to the directory containing tokenized category folders.
        output_path (str): Path to save the final train.bin and val.bin files.
        proportions (dict): A dictionary mapping category names to their percentage (e.g., {'RedPajamaBook': 0.2}).
        total_token_budget (int): The total number of tokens desired for the training set.
        val_tokens_per_category (int): The number of tokens to take from each category for the validation set.
    """
    base_path = Path(tokenized_data_path)
    output_base_path = Path(output_path)
    output_base_path.mkdir(parents=True, exist_ok=True)

    # Validate proportions
    if abs(sum(proportions.values()) - 1.0) > 1e-6:
        raise ValueError(f"Proportions must sum to 1.0, but they sum to {sum(proportions.values())}")

    # --- Create Validation Set ---
    logging.info("Creating validation set...")
    val_data = []
    for category in proportions.keys():
        bin_file = base_path / category / "train.bin"
        if not bin_file.exists():
            logging.warning(f"Validation: {bin_file} not found. Skipping.")
            continue
        
        # Memory-map the file to read tokens from the end
        m = np.memmap(bin_file, dtype=np.uint16, mode='r')
        # Take tokens from the end of the file for validation
        val_data.append(m[-val_tokens_per_category:])
        logging.info(f"  - Took {val_tokens_per_category:,} tokens from {category} for validation.")

    val_arr = np.concatenate(val_data)
    val_filename = output_base_path / 'val.bin'
    logging.info(f"Writing validation data to {val_filename} ({len(val_arr):,} tokens)...")
    val_arr.tofile(val_filename)
    logging.info("Validation set created successfully.")

    # --- Create Training Set ---
    logging.info("Creating training set...")
    train_filename = output_base_path / 'train.bin'
    # Create an empty file to append to
    with open(train_filename, 'wb') as f:
        pass

    total_tokens_written = 0
    for category, proportion in tqdm(proportions.items(), desc="Processing categories for training"):
        num_tokens_to_take = int(total_token_budget * proportion)
        
        bin_file = base_path / category / "train.bin"
        if not bin_file.exists():
            logging.warning(f"Training: {bin_file} not found. Skipping.")
            continue
        
        m = np.memmap(bin_file, dtype=np.uint16, mode='r')
        # We took the last `val_tokens_per_category` for validation, so don't reuse them
        train_data = m[:-val_tokens_per_category]

        if len(train_data) < num_tokens_to_take:
            logging.warning(f"  - Category {category} has only {len(train_data):,} tokens, but {num_tokens_to_take:,} were requested. Using all available tokens.")
            num_tokens_to_take = len(train_data)
        
        chunk = train_data[:num_tokens_to_take]

        # Append to the final train.bin
        with open(train_filename, 'ab') as f:
            f.write(chunk.tobytes())
        
        total_tokens_written += len(chunk)
        logging.info(f"  - Wrote {len(chunk):,} tokens from {category}.")

    logging.info(f"Training set created successfully at {train_filename}")
    logging.info(f"Total tokens in training set: {total_tokens_written:,}")
    return train_filename, val_filename


if __name__ == '__main__':
    # --- Configuration ---
    # STEP 1: Define the proportions for each category.
    # IMPORTANT: These values must sum up to 1.0
    category_proportions = {
        'RedPajamaCommonCrawl': 0.50,
        'RedPajamaC4':          0.20,
        'RedPajamaGithub':      0.10,
        'RedPajamaBook':        0.08,
        'RedPajamaArXiv':       0.06,
        'RedPajamaWikipedia':   0.04,
        'RedPajamaStackExchange': 0.02,
    }

    # STEP 2: Define your total token budget for the training set.
    TRAIN_BUDGET_TOKENS = 3_000_000_000

    # --- Paths ---
    tokenized_data_dir = '/data/yaxin/pretrain_data_analysis/data/slimpajama_tokenized_10pc'
    final_dataset_dir = '/data/yaxin/pretrain_data_analysis/data/slimpajama-3b'

    create_mixed_dataset(
        tokenized_data_path=tokenized_data_dir,
        output_path=final_dataset_dir,
        proportions=category_proportions,
        total_token_budget=TRAIN_BUDGET_TOKENS
    )
    logging.info("Dataset creation complete.")
