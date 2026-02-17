# config for training GPT-2 (124M) with DeepSpeed ZeRO-2
# ZeRO-2: Optimizer state + gradient partitioning
# Best for: Balanced memory savings and communication overhead
# launch: deepspeed --num_gpus=8 train_deepspeed.py config/train_gpt2_deepspeed_zero2.py
#
# To resume from latest checkpoint:
#   deepspeed --num_gpus=8 train_deepspeed.py config/train_gpt2_deepspeed_zero2.py --init_from=resume
# To resume from specific checkpoint:
#   deepspeed --num_gpus=8 train_deepspeed.py config/train_gpt2_deepspeed_zero2.py --init_from=resume --resume_from_checkpoint=ckpt_step_005000.pt

wandb_log = True
wandb_project = 'pretrain-data-analysis'
wandb_run_name = 'data/openwebtext_20pct_llmlingua2_compressed'

# Use the 20% subset dataset for data analysis experiments
dataset = '/data/yaxin/GRPO-Zero/rephrase_rl/openwebtext_20pct_our_rephraser'

# DeepSpeed configuration
deepspeed_config = 'deepspeed_configs/zero2_config.json'
use_deepspeed = True


# Checkpoint settings
# Checkpoints will be saved with step numbers: ckpt_step_001000.pt, ckpt_step_002000.pt, etc.
# A latest checkpoint (ckpt.pt) is also maintained for easy resuming
# resume_from_checkpoint = 'ckpt_step_005000.pt'  # uncomment to resume from specific checkpoint
# these make the total batch size be ~0.5M -> will be ~1M after change
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 36 # was 12
block_size = 1024 
gradient_accumulation_steps = 5 * 8

# Reduced iterations for 20% subset (original was 600k for full dataset)
# Since we have 20% of the data, we use 20% of the iterations to maintain similar data exposure
max_iters = 40000  # 20% of 600000
lr_decay_iters = 40000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# Disable compile for DeepSpeed compatibility
compile = False

# --- Overrides for experiment: balanced_mix ---
out_dir = '/data/yaxin/pretrain_data_analysis/out/gpt2-medium-balanced_mix'
dataset = 'slimpajama-3b-balanced_mix'
wandb_run_name = 'balanced_mix'
