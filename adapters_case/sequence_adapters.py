## Install:
# !pip install --upgrade pip
# !pip install transformers datasets evaluate peft python-dotenv trl

from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
from dotenv import load_dotenv
from glob import glob
from pathlib import Path
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
load_dotenv()

class InstructAdapter():
    def __init__(
            self,
            original_instruct_model,
            base_adapter_checkpoint,
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
            assistant_only_loss=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1, 
            target_modules=["q_proj", "v_proj"]
        ):
        self.original_instruct_model = original_instruct_model
        self.base_adapter_checkpoint = base_adapter_checkpoint
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
        self.assistant_only_loss = assistant_only_loss
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules

        # Maybe move the path handling outside
        if not output_dir:
            self.output_dir = Path(Path(__file__).parent).joinpath("output", "sequential")

        if assistant_only_loss:
            extra_args = {"assistant_only_loss": True}
            self.output_dir = Path(self.output_dir).joinpath("assistant_loss")
        else:
            extra_args = {"dataset_text_field": "text"}
            self.output_dir = Path(self.output_dir).joinpath("lm_loss")

        if self.add_prompt:
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

    def formatting_func_lm_loss(self, examples, tokenizer, add_prompt):
        if add_prompt:
            system_promp = {"role": "system", "content": self.sys_prompt}
            convos = examples["messages"]
            for convo in convos:
                convo.insert(0, system_promp)
        texts = tokenizer.apply_chat_template(examples["messages"], tokenize = False, add_generation_prompt = False)
        return { "text": texts }

    def formatting_func_assistant_loss(self, examples, tokenizer, add_prompt):
        convos = examples["messages"]
        if add_prompt:
            system_promp = {"role": "system", "content": "You are the philospher Socrates. Answer as he would with the knowlegde of his time and no modern one."}
            for convo in convos:
                convo.insert(0, system_promp)
        return { "messages": convos }

    def train(self):
        # Use original tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.original_instruct_model)

        # Add custom chat template
        if self.chat_template_path:
            tokenizer.chat_template = open(self.chat_template_path).read()

        # Get the original instruct model
        if self.attn_implementation:
            model = AutoModelForCausalLM.from_pretrained(self.original_instruct_model, attn_implementation=self.attn_implementation)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.original_instruct_model)

        # Insert & merge the base adapter
        ################# Insert as weighted #################
        model = PeftModel.from_pretrained(model, self.base_adapter_checkpoint, adapter_name="base")
        weights = [0.6]
        adapter_name = "weighted_base"
        model.add_weighted_adapter(["base"], weights, adapter_name)
        model.set_adapter(adapter_name)
        model.delete_adapter("base")

        model = model.merge_and_unload().to("cuda")
                
        # import dataset
        character_data = load_from_disk(self.datafile)

        formatting_func = self.formatting_func_lm_loss
        if self.assistant_only_loss:
            formatting_func = self.formatting_func_assistant_loss
        character_data = character_data.map(formatting_func, batched = True, fn_kwargs={"tokenizer":tokenizer, "add_prompt": self.add_prompt}, remove_columns=character_data.column_names, num_proc=self.num_proc)

        # Add LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules
        )
        
        model = get_peft_model(model, peft_config, adapter_name="character_adapter")

        # Train
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=character_data,
            args=SFTConfig(**self.sft_args)
        )

        trainer.train()
        print(f"Checkpoints saved to {self.output_dir}")

        # Use the last checkpoint to merge the model
        ### NOTE: Checkpoints are saved in case another one should be used for this step in a separate file
        model = model.merge_and_unload()
        model.save_pretrained(self.output_dir.joinpath("final_model"))

def check_digit(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ check_digit(c) for c in re.split(r'(\d+)', text) ]

if __name__ == "__main__":
    '''
    This script injects adapter 1 (obtained from fine-tuning the foundational model) into
    the instruct and produces a second adapter with the conversation data
    '''

    chat_template_path = "adapters_case/generate_tag_template.jinja"
    sys_prompt = "You are the philospher Socrates. Answer as he would with the knowlegde of his time and no modern one."

    # Get the foundational adapter
    checkpoints_path = "adapters_case/output" 
    base_adapter_checkpoint = [sorted(glob(f"{path}/*"), key=natural_keys)[-1] for path in glob(f"{checkpoints_path}/*") if "foundational_adapter" in path and Path(path).is_dir()][0]
    output_dir = "adapters_case/output/sequential"


    for assistant_only_loss in [False, True]:
        # Use both with and without prompt cases
        for add_prompt in [False, True]:
            adapter_trainer = InstructAdapter(
                "google/gemma-3-1b-it", # original_instruct_model
                base_adapter_checkpoint, # base_adapter_checkpoint
                datafile="test_data/train", # arrow
                output_dir=output_dir,
                num_train_epochs = 1,
                add_prompt=add_prompt,
                assistant_only_loss=assistant_only_loss,
                chat_template_path = chat_template_path,
                sys_prompt = sys_prompt
            )

            adapter_trainer.train()
