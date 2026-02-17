import os
from datasets import load_dataset
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sample_dataset(dataset_path, output_path, sample_percentage=0.1, seed=42):
    """
    Samples a percentage of a dataset and saves it.

    Args:
        dataset_path (str): Path to the dataset directory.
        output_path (str): Path to save the sampled dataset.
        sample_percentage (float): The percentage of the dataset to sample.
        seed (int): Random seed for reproducibility.
    """
    base_path = Path(dataset_path)
    output_base_path = Path(output_path)
    output_base_path.mkdir(parents=True, exist_ok=True)

    categories = [d.name for d in base_path.iterdir() if d.is_dir()]
    logging.info(f"Found categories: {categories}")

    for category in categories:
        logging.info(f"Processing category: {category}")
        category_path = base_path / category
        output_category_path = output_base_path / category
        output_category_path.mkdir(exist_ok=True)

        # Glob for all compressed jsonl files
        file_paths = [str(f) for f in category_path.glob('*.jsonl.zst')]
        if not file_paths:
            logging.warning(f"No '.jsonl.zst' files found in {category_path}")
            continue

        logging.info(f"Loading dataset for {category} from {len(file_paths)} files...")
        # Load the dataset from the list of files
        dataset = load_dataset('json', data_files=file_paths, split='train')
        
        # Sample the dataset
        logging.info(f"Sampling {sample_percentage * 100}% of the data for {category}...")
        num_samples = int(len(dataset) * sample_percentage)
        sampled_dataset = dataset.shuffle(seed=seed).select(range(num_samples))
        
        # Save the sampled dataset
        output_file = output_category_path / "sampled.jsonl.gz"
        logging.info(f"Saving sampled data for {category} to {output_file}...")
        sampled_dataset.to_json(output_file, compression="gzip")
        
        logging.info(f"Finished processing category: {category}")

if __name__ == '__main__':
    # Define paths - using absolute paths as requested
    slimpajama_path = '/data/yaxin/pretrain_data_analysis/data/SlimPajama-627B-DC/train'
    sampled_output_path = '/data/yaxin/pretrain_data_analysis/data/slimpajama_sampled_10pc'
    
    sample_dataset(slimpajama_path, sampled_output_path)
    logging.info("All categories have been sampled.")
