import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

def tokenize_dataset(dataset_path, output_path):
    """
    Tokenizes a dataset and saves it to binary files, counting tokens per category.

    Args:
        dataset_path (str): Path to the sampled dataset directory.
        output_path (str): Path to save the tokenized data.
    """
    base_path = Path(dataset_path)
    output_base_path = Path(output_path)
    output_base_path.mkdir(parents=True, exist_ok=True)

    categories = [d.name for d in base_path.iterdir() if d.is_dir()]
    logging.info(f"Found categories: {categories}")

    enc = tiktoken.get_encoding("gpt2")

    token_counts = {}

    for category in tqdm(categories, desc="Processing categories"):
        logging.info(f"Processing category: {category}")
        category_path = base_path / category
        output_category_path = output_base_path / category
        output_category_path.mkdir(exist_ok=True)

        file_path = category_path / "sampled.jsonl.gz"
        if not file_path.exists():
            logging.warning(f"File not found: {file_path}. Skipping category.")
            continue
        
        dataset = load_dataset('json', data_files=str(file_path), split='train')

        def process(example):
            ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
            ids.append(enc.eot_token) # add the end of text token
            out = {'ids': ids, 'len': len(ids)}
            return out

        logging.info(f"Tokenizing data for {category}...")
        tokenized = dataset.map(
            process,
            remove_columns=['text', 'meta'], # SlimPajama has 'text' and 'meta'
            desc=f"tokenizing {category}",
            num_proc=num_proc,
        )

        total_tokens = np.sum(tokenized['len'])
        token_counts[category] = total_tokens
        logging.info(f"Category {category} has {total_tokens:,} tokens.")

        # Save to binary file
        for split in ['train']: # We only have a single file per category
            arr_len = total_tokens
            filename = output_category_path / f'{split}.bin'
            dtype = np.uint16 # gpt2 vocab size is 50257, fits in uint16
            
            logging.info(f"Writing tokenized data to {filename}...")
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            
            idx = 0
            for example in tqdm(tokenized, desc=f"writing {filename}"):
                arr[idx : idx + example['len']] = example['ids']
                idx += example['len']
            arr.flush()

    logging.info("--- Token Counts Summary ---")
    for category, count in token_counts.items():
        logging.info(f"{category}: {count:,} tokens")
    total = sum(token_counts.values())
    logging.info(f"Total tokens: {total:,}")
    logging.info("--------------------------")

if __name__ == '__main__':
    sampled_data_path = '/data/yaxin/pretrain_data_analysis/data/slimpajama_sampled_10pc'
    tokenized_output_path = '/data/yaxin/pretrain_data_analysis/data/slimpajama_tokenized_10pc'
    tokenize_dataset(sampled_data_path, tokenized_output_path)
    logging.info("Tokenization complete.")
