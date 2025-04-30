NUM_GPUS = 4
MODEL_PATH = 'instructlab/granite-7b-lab'
MAX_BATCH_LENGTH = 400
MAX_SEQ_LEN = 10000

import torch 
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    get_scheduler,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
    StateDictType,
)
import os
import gc

# Initialize distributed environment
torch.distributed.init_process_group(backend='nccl')
local_rank = int(os.environ.get('LOCAL_RANK', 0))
torch.cuda.set_device(local_rank)

# Create process groups for hybrid sharding
world_size = torch.distributed.get_world_size()
hybrid_shard_size = world_size // 2  # Split into 2 groups of 2 GPUs each

# Create two process groups for hybrid sharding
group1 = torch.distributed.new_group(ranks=[0, 1])
group2 = torch.distributed.new_group(ranks=[2, 3])

if local_rank == 0:
    print(f"Initialized process groups for hybrid sharding")

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    cpu_offload=CPUOffload(offload_params=False),
    auto_wrap_policy=None,
    limit_all_gathers=True,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    process_group=(group1, group2),  # Pass both groups as a tuple
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

if local_rank == 0:
    print(f"Model and optimizer prepared on rank {local_rank}")

dummy_input = torch.randint(0, 1000, (1, 32), device=torch.cuda.current_device())
dummy_labels = torch.randint(0, 1000, (1, 32), device=torch.cuda.current_device())

step = 0
while True:
    try:
        outputs = model(dummy_input, labels=dummy_labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if local_rank == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
        save_dir = os.path.join('/tmp/checkpoints', f'checkpoint_{step}')
        os.makedirs(save_dir, exist_ok=True)

        if local_rank == 0:
            print(f"Saving checkpoint {step}")
    
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model_state = model.state_dict()
            torch.save(model_state, os.path.join(save_dir, 'model.pt'))
            if local_rank == 0:
                print(f"Saved model state {step}")

        optim_state = optimizer.state_dict()
        torch.save(optim_state, os.path.join(save_dir, 'optimizer.pt'))

        if local_rank == 0:
            print(f"Saved optimizer state {step}")
    
        del outputs
        del loss
        del model_state
        del optim_state
        torch.cuda.empty_cache()
        gc.collect()
        
        step += 1

    except KeyboardInterrupt:
        if local_rank == 0:
            print("Stopping checkpoint loop")
        break

