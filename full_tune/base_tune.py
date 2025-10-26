## Install:
# !pip install --upgrade pip
# !pip install transformers datasets evaluate peft python-dotenv

from datasets import load_dataset
from dotenv import load_dotenv
from functools import partial
from pathlib import Path
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
load_dotenv()

class BaseFineTune():
    def __init__(
            self,
            base_model,
            dataset_file,
            output_dir=None,
            attn_implementation='eager',
            num_proc=4,
            text_block_size = 256,
            learning_rate = 1e-5,
            weight_decay = 0.01,
            save_steps = 5000,
            num_train_epochs = 10,
            adapter_training=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1, 
            target_modules=["q_proj", "v_proj"]
        ):
        self.base_model = base_model
        self.dataset_file = dataset_file
        self.output_dir = Path(output_dir) if output_dir else None
        self.attn_implementation = attn_implementation
        self.num_proc = num_proc
        self.text_block_size = text_block_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_steps = save_steps
        self.num_train_epochs = num_train_epochs
        self.adapter_training = adapter_training
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules

        if not output_dir:
            self.output_dir = Path(Path(__file__).parent).joinpath("output", "base")
        if adapter_training:
            self.output_dir = self.output_dir.joinpath("foundational_adapter")
        
    def preprocess_function(self, tokenizer, examples):
        return tokenizer([f" {examples['sentence']}"]) # Some models' tokenizer might supress spaces

    def group_texts(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= self.text_block_size:
            total_length = (total_length // self.text_block_size) * self.text_block_size
        
        # split chunks
        result = {
            key: [val[i:i+self.text_block_size] for i in range(0, total_length, self.text_block_size)] for key, val in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def train(self):
        # import dataset
        character_data = load_dataset("json", split='train', data_files=self.dataset_file)

        # Get the model from HF
        if self.attn_implementation:
            model = AutoModelForCausalLM.from_pretrained(self.base_model, attn_implementation=self.attn_implementation)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.base_model)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        # preprocess data
        tokenized_soc = character_data.map(partial(self.preprocess_function, tokenizer), batched=True, num_proc=self.num_proc, remove_columns=character_data.column_names)
        
        # Concatenate all the token sequences & splits into chunks
        lm_dataset = tokenized_soc.map(self.group_texts, batched=True, num_proc=self.num_proc)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Train
        # Add LoRA if needed
        if self.adapter_training:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules
            )
            
            model = get_peft_model(model, peft_config)

        training_args = TrainingArguments(
            output_dir = self.output_dir,
            learning_rate = self.learning_rate,
            weight_decay = self.weight_decay,
            save_steps = self.save_steps,
            num_train_epochs = self.num_train_epochs
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_dataset,
            data_collator=data_collator,
            processing_class=tokenizer
        )

        trainer.train()
        print(f"saved to {self.output_dir}")

if __name__ == "__main__":
    base_trainer = BaseFineTune(
        base_model = "google/gemma-3-1b-pt",
        dataset_file = "/home/david/conv_data_generation/model_training/test_data/full_no_names.jsonl",
        text_block_size = 254,
        learning_rate = 1e-5,
        weight_decay = 0.01,
        save_steps = 5000,
        num_train_epochs = 1,
        num_proc = 2
    )

    base_trainer.train()
