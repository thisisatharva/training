from instructlab.training import run_training, TrainingArgs, TorchrunArgs, FSDPOptions

# MODIFY THESE VALUES
NUM_GPUS = 8
MODEL_PATH =  'instructlab/granite-7b-lab'
DATA_PATH = 'sample-data/train_all_pruned_SDG.jsonl'
MAX_BATCH_LENGTH = 13000 
MAX_SEQ_LEN = 10000

def phase_10():
    torch_args = TorchrunArgs(
        nproc_per_node=NUM_GPUS,
        nnodes=1,
        node_rank=0,
        rdzv_id=123,
        rdzv_endpoint='0.0.0.0:1738'
    )
    training_args = TrainingArgs(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        ckpt_output_dir='/tmp/checkpoints',
        data_output_dir='/dev/shm',
        max_seq_len=MAX_SEQ_LEN,
        max_batch_len=MAX_BATCH_LENGTH,
        num_epochs=10,
        warmup_steps=25,
        learning_rate=6e-6,
        save_samples=0,
        effective_batch_size=3840,
        accelerate_full_state_at_epoch=True,
        checkpoint_at_epoch=True,
        use_liger=False,
        distributed_backend='fsdp',
        disable_flash_attn=True,
        fsdp_options=FSDPOptions(
                cpu_offload_params=True,
        )
        
    )
    run_training(torch_args, training_args)
    
if __name__ == '__main__':
    print("Starting training...")
    phase_10()