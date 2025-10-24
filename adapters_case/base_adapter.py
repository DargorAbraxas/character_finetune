from ..full_tune.base_tune import BaseFineTune

if __name__ == "__main__":
    base_trainer = BaseFineTune(
        base_model = "google/gemma-3-1b-pt",
        datafile = "/work/gemma_socrates_v0/full_no_names.jsonl",
        output_dir = "/work/gemma_socrates_v0/output_adapter",
        text_block_size = 256,
        learning_rate = 1e-5,
        weight_decay = 0.01,
        save_steps = 5000,
        num_train_epochs = 10,
        adapter_training=True,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1, 
        target_modules=["q_proj", "v_proj"]
    )
    base_trainer.train()
