import os
import subprocess
from pathlib import Path
import logging
import shutil

# Import the function from our data creation script
from create_finetune_data import create_mixed_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Configuration ---
# A list of experiments to run. Each experiment is a dictionary with a unique name
# and a set of proportions for the data mix.
# IMPORTANT: Proportions for each experiment MUST sum to 1.0
experiments = [
    # {
    #     'name': 'balanced_mix',
    #     'proportions': {
    #         'RedPajamaCommonCrawl': 0.15,
    #         'RedPajamaC4':          0.15,
    #         'RedPajamaGithub':      0.15,
    #         'RedPajamaBook':        0.15,
    #         'RedPajamaArXiv':       0.15,
    #         'RedPajamaWikipedia':   0.15,
    #         'RedPajamaStackExchange': 0.10,
    #     }
    # },
    # {
    #     'name': 'commoncrawl_heavy',
    #     'proportions': {
    #         'RedPajamaCommonCrawl': 0.70,
    #         'RedPajamaC4':          0.10,
    #         'RedPajamaGithub':      0.05,
    #         'RedPajamaBook':        0.05,
    #         'RedPajamaArXiv':       0.05,
    #         'RedPajamaWikipedia':   0.03,
    #         'RedPajamaStackExchange': 0.02,
    #     }
    # },
    # {
    #     'name': 'academic_focused',
    #     'proportions': {
    #         'RedPajamaArXiv':       0.40,
    #         'RedPajamaWikipedia':   0.30,
    #         'RedPajamaBook':        0.10,
    #         'RedPajamaC4':          0.10,
    #         'RedPajamaCommonCrawl': 0.05,
    #         'RedPajamaGithub':      0.03,
    #         'RedPajamaStackExchange': 0.02,
    #     }
    # },
    # {
    #     'name': 'code_focused',
    #     'proportions': {
    #         'RedPajamaGithub':      0.50,
    #         'RedPajamaStackExchange': 0.30,
    #         'RedPajamaCommonCrawl': 0.10,
    #         'RedPajamaC4':          0.05,
    #         'RedPajamaArXiv':       0.03,
    #         'RedPajamaBook':        0.01,
    #         'RedPajamaWikipedia':   0.01,
    #     }
    # },
    # {
    #     'name': 'narrative_focused',
    #     'proportions': {
    #         'RedPajamaBook':        0.40,
    #         'RedPajamaC4':          0.30,
    #         'RedPajamaCommonCrawl': 0.20,
    #         'RedPajamaWikipedia':   0.05,
    #         'RedPajamaArXiv':       0.02,
    #         'RedPajamaGithub':      0.02,
    #         'RedPajamaStackExchange': 0.01,
    #     }
    # },
    {
        'name': 'communication_focused',
        'proportions': {
            'RedPajamaStackExchange': 0.50,
            'RedPajamaCommonCrawl': 0.30,
            'RedPajamaC4':          0.10,
            'RedPajamaBook':        0.05,
            'RedPajamaArXiv':       0.03,
            'RedPajamaGithub':      0.01,
            'RedPajamaWikipedia':   0.01,
        }
    },
    {
        'name': 'wikipedia_focused',
        'proportions': {
            'RedPajamaWikipedia': 0.50,
            'RedPajamaCommonCrawl': 0.30,
            'RedPajamaC4':          0.10,
            'RedPajamaBook':        0.05,
            'RedPajamaArXiv':       0.03,
            'RedPajamaGithub':      0.01,
            'RedPajamaStackExchange': 0.01,
        }
    },
]

# --- Paths and Settings ---
# Path to the 10% tokenized dataset
TOKENIZED_DATA_DIR = '/data/yaxin/pretrain_data_analysis/data/slimpajama_tokenized_10pc'
# Base directory where final mixed datasets will be stored
FINAL_DATASETS_BASE_DIR = '/data/yaxin/pretrain_data_analysis/data'
# Base directory for model checkpoints and outputs
MODEL_OUTPUT_BASE_DIR = '/data/yaxin/pretrain_data_analysis/out'
# Base config file to use for training
BASE_CONFIG_FILE = 'config/train_gpt2_deepspeed_zero2.py'
# Total token budget for the training set
TRAIN_BUDGET_TOKENS = 3_000_000_000
# Number of GPUs to use for training
# NUM_GPUS = 4 # This will be determined by the length of GPU_IDS
# Specify the exact GPU IDs to use
GPU_IDS = "4,5,6,7"


def run_command(command):
    """Executes a command and logs its output."""
    logging.info(f"Executing command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(process.stdout.readline, ''):
        logging.info(line.strip())
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")

def main():
    """Main function to run the experiment pipeline."""
    for exp in experiments:
        exp_name = exp['name']
        proportions = exp['proportions']
        logging.info(f"--- Starting Experiment: {exp_name} ---")

        # 1. Define paths for this experiment
        exp_data_dir = Path(FINAL_DATASETS_BASE_DIR) / f"slimpajama-3b-{exp_name}"
        exp_model_out_dir = Path(MODEL_OUTPUT_BASE_DIR) / f"gpt2-medium-{exp_name}"
        exp_config_dir = Path('./config') / 'generated'
        exp_config_file = exp_config_dir / f"config_{exp_name}.py"

        # Create directories
        exp_data_dir.mkdir(exist_ok=True)
        exp_model_out_dir.mkdir(exist_ok=True)
        exp_config_dir.mkdir(exist_ok=True)

        # 2. Create the dataset for the current mix
        logging.info(f"Creating dataset for '{exp_name}' at {exp_data_dir}...")
        create_mixed_dataset(
            tokenized_data_path=TOKENIZED_DATA_DIR,
            output_path=str(exp_data_dir),
            proportions=proportions,
            total_token_budget=TRAIN_BUDGET_TOKENS
        )
        logging.info(f"Dataset for '{exp_name}' created successfully.")

        # 3. Generate the experiment-specific config file
        logging.info(f"Generating config file: {exp_config_file}")
        with open(BASE_CONFIG_FILE, 'r') as f:
            base_config_content = f.read()
        
        # Override the necessary parameters
        # Note: dataset path is relative to the `data/` directory in `train_deepspeed.py`
        dataset_name_for_config = exp_data_dir.name
        
        override_config = f"""
# --- Overrides for experiment: {exp_name} ---
out_dir = '{exp_model_out_dir}'
dataset = '{dataset_name_for_config}'
wandb_run_name = '{exp_name}'
"""
        
        full_config_content = base_config_content + override_config
        with open(exp_config_file, 'w') as f:
            f.write(full_config_content)
        logging.info("Config file generated.")

        # 4. Launch the training process
        logging.info(f"Launching DeepSpeed training for '{exp_name}'...")
        deepspeed_command = [
            'deepspeed',
            f'--include=localhost:{GPU_IDS}',
            'train_deepspeed.py',
            str(exp_config_file)
        ]
        
        try:
            run_command(deepspeed_command)
            logging.info(f"--- Experiment '{exp_name}' completed successfully! ---")
        except RuntimeError as e:
            logging.error(f"--- Experiment '{exp_name}' failed! ---")
            logging.error(e)
            # Decide if you want to stop or continue with the next experiment
            # break # Uncomment to stop on failure
            continue # Comment out to stop on failure
    
    logging.info("All experiments have been processed.")

if __name__ == '__main__':
    main()
