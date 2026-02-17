# Config for training GPT-2 (124M) with DeepSpeed pipeline parallelism
# Example launch (2 pipeline stages on 4 GPUs):
#   deepspeed --num_gpus=4 train_deepspeed.py config/train_gpt2_pipeline.py

wandb_log = True
wandb_project = 'parrella-gpt2'
wandb_run_name = 'gpt2-124M-pipeline'

use_deepspeed = True
deepspeed_config = 'deepspeed_configs/pipeline_config.json'

# Enable pipeline parallelism with two uniform stages.
# You can set this to "auto" in the JSON config, but overriding here keeps it explicit.
pipeline_parallel_size = 2
pipeline_partition_method = 'uniform'
pipeline_activation_checkpoint_interval = 0

# Effective batch: 12 micro * 1024 tokens * 5 grad acc * 4 GPUs = 245,760 tokens
batch_size = 36
block_size = 1024
gradient_accumulation_steps = 5 * 4

max_iters = 350
lr_decay_iters = 350

eval_interval = 100
eval_iters = 100
log_interval = 10

weight_decay = 1e-1

# Compilation is disabled automatically when using DeepSpeed, but keep it explicit
compile = False
