# Project: Nanochat Hyperparameter Sweep & Predictability Analysis

## Objective
Modify this fork of `karpathy/nanochat` to support automated Weights & Biases (W&B) hyperparameter sweeps, extract deep network statistics (weight and gradient norms) during training, and save early-stage checkpoints. The goal is to collect early training trajectory data to predict final model performance (e.g., CORE score, validation bpb). 

The modified code must remain capable of running a "tiny" test mode locally on CPU/MPS, while being optimized for remote deployment on an 8xH100 node via `torchrun`.

## Task 1: Decouple the `--depth` Scaling Laws
By default, `nanochat` derives model width, heads, and learning rate strictly from the `--depth` argument. Update the main configuration parser (likely in `train.py` or `scripts/pretrain.py`) to accept override arguments for sweeping, while falling back to the default scaling laws if overrides are not provided.

1. **Add to `argparse`:**
   - `--depth` (type: int, default: 12)
   - `--n_embd_override` (type: int, default: None)
   - `--lr_override` (type: float, default: None)

2. **Update Config Logic:**
   ```python
   n_layer = args.depth
   # Override auto-calculated width if provided by the sweep
   n_embd = args.n_embd_override if args.n_embd_override is not None else calculate_optimal_width(n_layer)
   # Standard head dimension in nanochat is typically 64
   n_head = n_embd // 64 
   
   # Override learning rate if provided by the sweep
   learning_rate = args.lr_override if args.lr_override is not None else calculate_optimal_lr(n_layer)

```

## Task 2: Track Weight and Gradient Statistics

We need to track the L2 norm of the model's weights and gradients.

1. **Add this helper function near the top of the training script:**
```python
import torch

def compute_network_stats(model):
    """Calculates L2 norm of weights and gradients."""
    weight_norm = sum(p.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
    grad_norm = sum(p.grad.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    return weight_norm, grad_norm

```


2. **Inject into the Training Loop:**
Locate the training step (where `loss.backward()` and `optimizer.step()` occur). Calculate the stats immediately after the backward pass, before the optimizer steps or zeros the gradients.
```python
loss.backward()

# Calculate stats before clipping or stepping
weight_norm, grad_norm = compute_network_stats(model)

optimizer.step()

```


3. **Update W&B Logging:**
Find the dictionary passed to `wandb.log()` and add the new metrics.
```python
if wandb_log and step % log_interval == 0:
    wandb.log({
        "step": step,
        "train/loss": loss.item(),
        "train/weight_norm": weight_norm,
        "train/grad_norm": grad_norm,
        "lr": current_lr,
        # ... (keep existing metrics like val_bpb)
    })

```



## Task 3: Implement Early Checkpointing

To save disk space and support early predictability analysis, modify the checkpointing logic to save the model state only at specific, early steps.

Find the checkpoint saving block and replace/update it with:

```python
early_save_steps = {250, 500, 1000}

if step in early_save_steps:
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'config': config_args, # Ensure the overridden config is saved
    }
    # Ensure out_dir is defined and exists
    torch.save(checkpoint, os.path.join(out_dir, f"ckpt_early_step_{step}.pt"))

```

## Task 4: Create the W&B Sweep Configuration File

Create a new file named `sweep.yaml` in the root directory of the repository.

```yaml
program: train.py # UPDATE THIS if the main script is inside scripts/
method: random
metric:
  name: val_bpb
  goal: minimize
parameters:
  depth:
    values: [8, 12, 16]
  n_embd_override:
    values: [256, 384, 512, 768]
  lr_override:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.01
command:
  - torchrun
  - --standalone
  - --nproc_per_node=8 # Remote H100 config; will be bypassed during local testing
  - ${program}
  - ${args}

```

## Task 5: Testing and Deployment Instructions (For the Human)

**1. Local Verification (Laptop)**
Run a tiny iteration on CPU to ensure the logging, stats calculation, and W&B integration work without CUDA errors.

```bash
python train.py --depth 2 --n_embd_override 64 --device cpu --max-iterations 10

```

**2. Remote Deployment (8xH100)**
Once pushed to the remote server, execute the sweep using `tmux` for minimal terminal interaction.

```bash
tmux new -s sweep_session
wandb sweep sweep.yaml
# Copy the generated sweep ID and start the agent:
wandb agent <USERNAME>/<PROJECT>/<SWEEP_ID>
# Detach from tmux using Ctrl+B, then D
```