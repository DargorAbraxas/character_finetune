
import sys
sys.path.insert(1, './')

from full_tune.base_tune import BaseFineTune

"""
This creates the adapter from the foundational model
"""

if __name__ == "__main__":
    base_trainer = BaseFineTune(
        base_model = "google/gemma-3-1b-pt",
        dataset_file = "test_data/full_no_names.jsonl",
        output_dir = "adapters_case/output",
        text_block_size = 254,
        learning_rate = 1e-5,
        weight_decay = 0.01,
        save_steps = 5000,
        num_train_epochs = 1,
        num_proc = 2,
        adapter_training=True,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1, 
        target_modules=["q_proj", "v_proj"]
    )
    base_trainer.train()
