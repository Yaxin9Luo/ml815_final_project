# config for training GPT-2 (124M) with DeepSpeed ZeRO-2
# Train on 1B tokens from SlimPajama balanced dataset (~3B tokens total)
# ZeRO-2: Optimizer state + gradient partitioning
# launch: deepspeed --num_gpus=8 train_deepspeed.py config/train_gpt2_deepspeed_zero2.py
#
# To resume from latest checkpoint:
#   deepspeed --num_gpus=8 train_deepspeed.py config/train_gpt2_deepspeed_zero2.py --init_from=resume
# To resume from specific checkpoint:
#   deepspeed --num_gpus=8 train_deepspeed.py config/train_gpt2_deepspeed_zero2.py --init_from=resume --resume_from_checkpoint=ckpt_step_001000.pt

wandb_log = True
wandb_project = 'slimpajama-gpt2-experiments'
wandb_run_name = 'gpt2-124m-balanced-120B'

# Dataset: SlimPajama balanced mixture
dataset = 'slimpajama-3b-experiments/slimpajama-3b-balanced'

# DeepSpeed configuration
deepspeed_config = 'deepspeed_configs/zero2_config.json'
use_deepspeed = True

# Output directory
out_dir = 'out/gpt2_124m_balanced_120B'

# Batch settings targeting ~0.5M tokens per optimizer step
# tokens_per_iter = batch_size * gradient_accumulation_steps * num_gpus * block_size
# = 48 * 5 * 8 * 1024 = 1,966,080 tokens/iter
batch_size = 48  # micro-batch per GPU
block_size = 1024
gradient_accumulation_steps = 5  # per-GPU (DeepSpeed does NOT divide by world_size)

# Training iterations for 120B tokens
max_iters = 61050
lr_decay_iters = 61050

# Learning rate schedule
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 200
decay_lr = True

# Eval and checkpoint settings
eval_interval = 200
eval_iters = 200
log_interval = 10

# Optimizer
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Disable compile for DeepSpeed compatibility
compile = False
