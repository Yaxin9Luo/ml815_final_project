# config for training GPT-2 Medium (350M) with DeepSpeed ZeRO-2 on SlimPajama experiments
# ZeRO-2: Optimizer state + gradient partitioning
# Best for: Balanced memory savings and communication overhead
# launch: deepspeed --num_gpus=4 train_deepspeed.py config/train_gpt2_medium_slimpajama.py

# To resume from latest checkpoint:
#   deepspeed --num_gpus=4 train_deepspeed.py config/train_gpt2_medium_slimpajama.py --init_from=resume
# To resume from specific checkpoint:
#   deepspeed --num_gpus=4 train_deepspeed.py config/train_gpt2_medium_slimpajama.py --init_from=resume --resume_from_checkpoint=ckpt_step_005000.pt

wandb_log = True
wandb_project = 'slimpajama-gpt2-medium-experiments'
wandb_run_name = 'balanced_proportions'  # Change this for different experiments

# Dataset configuration - update this for different experiments
dataset = 'slimpajama_experiments/balanced'  # Change to: balanced, web_heavy, academic_heavy, code_heavy, wikipedia_only

# DeepSpeed configuration
deepspeed_config = 'deepspeed_configs/zero2_config.json'
use_deepspeed = True

# Model configuration for GPT-2 Medium (350M parameters)
n_layer = 24
n_head = 16
n_embd = 1024

# Checkpoint settings
# Checkpoints will be saved with step numbers: ckpt_step_001000.pt, ckpt_step_002000.pt, etc.
# A latest checkpoint (ckpt.pt) is also maintained for easy resuming
# resume_from_checkpoint = 'ckpt_step_005000.pt'  # uncomment to resume from specific checkpoint

# Training configuration for 3B token dataset
# Adjusted for 4 GPUs and GPT-2 Medium model
batch_size = 8  # Reduced from 12 for larger model
block_size = 1024 
gradient_accumulation_steps = 4 * 4  # 4 GPUs * 4 accumulation steps = 16 effective batch size

# Training iterations - adjusted for 3B tokens
# With batch_size=8, gradient_accumulation_steps=16, block_size=1024:
# tokens_per_iter = 8 * 16 * 1024 = 131,072 tokens per iteration
# For 3B tokens: max_iters = 3,000,000,000 / 131,072 ≈ 22,900 iterations
max_iters = 25000  # Slightly more to ensure full dataset coverage
lr_decay_iters = 25000

# eval stuff
eval_interval = 500  # More frequent evaluation for experiments
eval_iters = 200
log_interval = 10

# Learning rate and optimization
learning_rate = 6e-4  # Standard for GPT-2 Medium
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay settings
decay_lr = True
warmup_iters = 2000
min_lr = 6e-5

# Disable compile for DeepSpeed compatibility
compile = False

# Output directory for this experiment
out_dir = 'out/gpt2_medium_slimpajama_experiments'

# --- Overrides for experiment: web_heavy ---
out_dir = 'out/gpt2_medium_web_heavy'
dataset = 'slimpajama-3b-web_heavy'
wandb_run_name = 'web_heavy'
