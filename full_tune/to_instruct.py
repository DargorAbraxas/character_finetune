## Install:
# !pip install --upgrade pip
# !pip install transformers datasets evaluate peft python-dotenv trl

from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
from dotenv import load_dotenv
from glob import glob
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
load_dotenv()

class InstructFineTune():
    def __init__(
            self,
            original_base_model,
            base_model_checkpoint,
            datafile,
            output_dir=None,
            add_prompt=False,
            sys_prompt=None,
            chat_template_path=None,
            attn_implementation='eager',
            learning_rate = 1e-5,
            weight_decay = 0.01,
            save_steps = 5000,
            num_train_epochs = 10,
            num_proc=4,
            assistant_only_loss=False
        ):
        self.original_base_model = original_base_model
        self.model_checkpoint = base_model_checkpoint
        self.datafile = datafile
        self.output_dir = output_dir
        self.add_prompt = add_prompt
        self.sys_prompt = sys_prompt
        self.chat_template_path = chat_template_path
        self.attn_implementation = attn_implementation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_steps = save_steps
        self.num_train_epochs = num_train_epochs
        self.num_proc = num_proc

        # Maybe move the path handling outside
        if not output_dir:
            self.output_dir = Path(Path(__file__).parent).joinpath("output")

        if assistant_only_loss:
            extra_args = {"assistant_only_loss": True}
            self.output_dir = Path(self.output_dir).joinpath("assistant_loss")
        else:
            extra_args = {"dataset_text_field": "text"}
            self.output_dir = Path(self.output_dir).joinpath("lm_loss")

        if add_prompt:
            self.output_dir = self.output_dir.parent.joinpath(f"{self.output_dir.name}_with_prompt")
        else:
            self.output_dir = self.output_dir.parent.joinpath(f"{self.output_dir.name}_no_prompt")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training config
        self.sft_args = {
            "output_dir": self.output_dir,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "save_steps": self.save_steps,
            "num_train_epochs": self.num_train_epochs,
            "dataset_num_proc": self.num_proc,
            **extra_args
        }

    def formatting_func(self, examples, tokenizer, add_prompt):
        if add_prompt:
            system_promp = {"role": "system", "content": self.sys_prompt}
            convos = examples["messages"]
            for convo in convos:
                convo.insert(0, system_promp)
        texts = tokenizer.apply_chat_template(examples["messages"], tokenize = False, add_generation_prompt = False)
        return { "text": texts }

    def train(self):
        # Use original tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.original_base_model)

        # Add custom chat template
        if self.chat_template_path:
            tokenizer.chat_template = open(self.chat_template_path).read()

        # Get the trained model
        if self.attn_implementation:
            model = AutoModelForCausalLM.from_pretrained(self.model_checkpoint, attn_implementation=self.attn_implementation)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_checkpoint)
                
        # import dataset
        character_data = load_from_disk(self.datafile)
        character_data = character_data.map(self.formatting_func, batched = True, fn_kwargs={"tokenizer":tokenizer, "add_prompt": self.add_prompt}, remove_columns=character_data.column_names, num_proc=self.num_proc)

        # Train
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=character_data,
            args=SFTConfig(**self.sft_args)
        )

        trainer.train()
        print(f"saved to {self.output_dir}")

def check_digit(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ check_digit(c) for c in re.split(r'(\d+)', text) ]

if __name__ == "__main__":
    sys_prompt = "You are the philospher Socrates. Answer as he would with the knowlegde of his time and no modern one."
    chat_template_path = "full_tune/generate_tag_template.jinja"

    checkpoints_path = "full_tune/output"
    base_adapter_checkpoint = [sorted(glob(f"{path}/*"), key=natural_keys)[-1] for path in glob(f"{checkpoints_path}/*") if "base" in path and Path(path).is_dir()][0]

    # Use both with and without prompt cases
    for add_prompt in [False, True]:
        inst_trainer = InstructFineTune(
            original_base_model = "google/gemma-3-1b-pt",
            base_model_checkpoint = base_adapter_checkpoint,
            datafile = "test_data/train",
            add_prompt = add_prompt,
            chat_template_path = chat_template_path,
            sys_prompt = sys_prompt
        )

        inst_trainer.train()
