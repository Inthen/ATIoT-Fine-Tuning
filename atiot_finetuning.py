# -*- coding: utf-8 -*-

!apt-get update
!apt-get install cuda-toolkit-11-8
import os
os.environ["LD_LIBRARY_PATH"] += ":" + "/usr/local/cuda-11/lib64"
os.environ["LD_LIBRARY_PATH"] += ":" + "/usr/local/cuda-11.8/lib64"

!pip install datasets
!pip install trl
!pip install peft

!pip install bitsandbytes

!python -m bitsandbytes

from transformers.utils.import_utils import is_accelerate_available, is_bitsandbytes_available
is_bitsandbytes_available()

import os
import torch
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

# Model from Hugging Face hub
base_model = "NousResearch/Llama-2-7b-chat-hf"

login(token="add your Hugging Face login token here")

# Import ATIoT dataset from Hugging Face
data_files = {"train": "dataset_llm_random_train_small_attacks.jsonl", "test": "dataset_llm_random_test_small_attacks.jsonl"}
dataset = load_dataset("Inthen/ATIoT", data_files=data_files)

# Fine-tuned model
new_model = "llama-2-7b-chat-ATIoT-AttacksOnly"

compute_dtype = getattr(torch, "float16")

# Quantization configuration parameters
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Parameter-efficient fine-tuning parameters
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
# Training arguments
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=500,
    tokenizer=tokenizer,
    args=training_params,
    packing=True,
)

torch.cuda.empty_cache()

# Train the model
trainer.train()

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

from tensorboard import notebook
log_dir = "results/runs"
notebook.start("--logdir {} --port 4000".format(log_dir))
logging.set_verbosity(logging.CRITICAL)

torch.cuda.empty_cache()
testfile = open('/content/drive/MyDrive/LLM-Tests/dataset_llm_random_test_smallest_attacks.jsonl', 'r')
total_attacks = 28
true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0


# Prompt the fine-tuned model and compare answers with those expected from the training set.
while True:
    resultsfile = open('/content/drive/MyDrive/LLM-Tests/llm_finetuned_answers_2epochs.txt', 'a+')
    totalsfile = open('/content/drive/MyDrive/LLM-Tests/results_2epochs.txt', 'a+')
    torch.cuda.empty_cache()
    line = testfile.readline()
    if not line:
        break
    lis = line.split("#")
    question = lis[0]
    answer = lis[1]

    prompt = question
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    resultsfile.write(result[0]['generated_text'])
    print(result[0]['generated_text'])
    resultsfile.close()
    attacks = answer.split(",  ")

    split_response = result[0]['generated_text']
    response = split_response[3].split(",  ")

    for m in response:
        if m not in attacks:
            false_positives += 1
        else:
            true_positives += 1

    for m in attacks:
        if m not in response:
            false_negatives += 1

    true_negatives += abs((total_attacks -len(response)) - (total_attacks -len(attacks)))
    totalsfile.write('TP - ' + str(true_positives) + ' TN - ' + str(true_negatives) + ' FP - ' + str(false_positives) + ' FN - ' + str(false_negatives))
    totalsfile.close()
testfile.close()
resultsfile.close()
totalsfile.close()
print(true_positives)
print(true_negatives)
print(false_positives)
print(false_negatives)

torch.cuda.empty_cache()
