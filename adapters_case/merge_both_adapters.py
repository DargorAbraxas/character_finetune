## Install:
# !pip install --upgrade pip
# !pip install transformers datasets evaluate peft python-dotenv trl unidecode

from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
load_dotenv()

class CharacterAdapter():
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
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules

        if not output_dir:
            self.output_dir = Path(Path(__file__).parent).joinpath("output_adapter")

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
        tokenizer = AutoTokenizer.from_pretrained(self.original_instruct_model)

        # Add custom chat template
        if self.chat_template_path:
            tokenizer.chat_template = open(self.chat_template_path).read()

        # Get the original instruct model
        if self.attn_implementation:
            model = AutoModelForCausalLM.from_pretrained(self.original_instruct_model, attn_implementation=self.attn_implementation)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.original_instruct_model)

        # import dataset
        charatecter_data = load_from_disk(self.datafile)
        charatecter_data = charatecter_data.map(self.formatting_func, batched = True, fn_kwargs={"tokenizer":tokenizer, "add_prompt": self.add_prompt}, remove_columns=charatecter_data.column_names, num_proc=4)

        # Train
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules
        )

        model = get_peft_model(model, peft_config, adapter_name="character_adapter")

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=charatecter_data,
            args=SFTConfig(**self.sft_args)
        )

        trainer.train()
        print(f"Checkpoints saved to {self.output_dir}")

        # Insert the base adapter
        _ = model.load_adapter(self.base_adapter_checkpoint, adapter_name="base")
        # merge using TIES
        adapters = ["base", "character_adapter"]
        weights = [0.8, 2.0]
        adapter_name = "merged"
        model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="ties", density=0.8)
        model.set_adapter(adapter_name)
        model.delete_adapter("character_adapter")

        model = model.merge_and_unload()
        model.save_pretrained(self.output_dir.joinpath("final_model"))


if __name__ == "__main__":
    '''
    This script creates an adapter from the instruct model and makes a weighted merge of both to produce a merged model
    '''

    chat_template_path = "/work/gemma_socrates_v0/plus_to_inst_assistant/generate_tag_template.jinja"
    sys_prompt = "You are the philospher Socrates. Answer as he would with the knowlegde of his time and no modern one."

    # Use both with and without prompt cases
    for add_prompt in [False, True]:
        output_dir = "/work/gemma_socrates_v0/adapters_case/adapter_lm/new_output"
        if add_prompt:
            output_dir = "/work/gemma_socrates_v0/adapters_case/adapter_lm/output_prompt_o_proj"

        character_adapter = CharacterAdapter(
            "google/gemma-3-1b-it", # original_instruct_model
            "/work/gemma_socrates_v0/base_to_plus/output/checkpoint-3650/", # base_adapter_checkpoint
            output_dir, # datafile
            add_prompt=add_prompt,
            chat_template_path = chat_template_path,
            sys_prompt = sys_prompt
        )
        character_adapter.train()
