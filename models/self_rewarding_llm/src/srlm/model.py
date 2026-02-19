import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

def get_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    return bnb_config

def load_model(tokenizer_name, model_name):
    # bnb_config = get_bnb_config()

#     tokenizer = FastLanguageModel.from_pretrained(tokenizer_name, trust_remote_code=True)
#     tokenizer.chat_template = """<|begin_of_text|>
# {%- for message in messages %}
# {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
# {{- message['content'] + '<|eot_id|>' }}
# {%- endfor %}
# {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"""
#     tokenizer.eos_token = "<|eot_id|>"
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"

    # The tokenizer will automatically load by itself

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length = 2048,
        load_in_4bit = True,
        dtype = None
    )
    # model.config.pretraining_tp = 1

    tokenizer = get_chat_template(
        tokenizer,
        mapping={
            "user": "user", 
            "assistant": "assistant",
            "role": "role",  # Add this line for system messages
            "content": "content"
        },
        chat_template="chatml",
    )

    return model, tokenizer


# this is the lora / q lora
def create_peft_model(model):
    lora_dropout=0.1
    lora_alpha=16
    lora_r=64

    # model = prepare_model_for_kbit_training(model)
    model = FastLanguageModel.get_peft_model(
        model, 
        r = lora_r,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        bias = "none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        loftq_config=None,
        use_rslora=False   
    )

    model.print_trainable_parameters()

    return model