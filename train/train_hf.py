import argparse
import os
import sys
from pathlib import Path
from typing import Any, get_type_hints

import tqdm
from accelerate import Accelerator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn.functional as F
import wandb
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, Trainer, DataCollatorForSeq2Seq, \
    BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.append(str(Path(__file__).parent.parent))
from data.process_data import get_data_detect_ai, get_data
from eval.init import BaseArgs


class TrainArgs(BaseArgs):
    learning_rate: float = 2e-4
    config_path: str = "config/summary2.yml"

    def __init__(self):
        super().__init__()


def load_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation=args.attn_implementation,
        # device_map={"": int(os.environ.get("LOCAL_RANK") or 0)} if args.deepspeed or args.ddp else "auto",
        device_map=args.device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ) if args.load_in_4bit else None,
        torch_dtype=torch.bfloat16 if not args.load_in_4bit else None,
        trust_remote_code=True,
    )
    peft_config = LoraConfig(
        r=args.lora_config['r'],
        lora_alpha=args.lora_config['lora_alpha'],
        lora_dropout=args.lora_config['lora_dropout'],
        bias=args.lora_config['bias'],
        target_modules=args.lora_config['target_modules'],
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, shuffle=True):
        self.data = data
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data["input_ids"]))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        if self.shuffle:
            idx = self.indices[idx]
        return {
            "input_ids": self.data["input_ids"][idx],
            "labels": self.data["labels"][idx],
        }


def generate_data(
        dataset,
        tokonizer,
        max_length,
):
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    for data in tqdm.tqdm(dataset, total=len(dataset)):
        ids = tokenizer(data['input']).input_ids
        labels = tokenizer(data['output'] + tokenizer.eos_token).input_ids
        input_ids = ids + labels
        if len(input_ids) > max_length:
            continue
        attention_mask = [1] * len(input_ids)
        label_ids = [-100] * len(ids) + labels

        input_ids = torch.tensor(input_ids + [tokonizer.eos_token_id] * (max_length - len(input_ids)))
        attention_mask = torch.tensor(attention_mask + [0] * (max_length - len(attention_mask)))
        label_ids = torch.tensor(label_ids + [-100] * (max_length - len(label_ids)))

        # Shift labels
        # input_ids = input_ids[:-1]
        # label_ids = label_ids[1:]

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(label_ids)
    print(f"-----num_of_dataset:{len(all_input_ids)}----------")
    return {
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_mask),
        "labels": torch.stack(all_labels),
    }


if __name__ == "__main__":
    args = TrainArgs()
    args.parse_args()
    print("???", args.model_path)
    print("???", args.config_path)
    print("???", args.save_dir)
    if args.ddp:
        args.device_map = {"": Accelerator().process_index}
    else:
        args.device_map = 'auto'
    # dataset = get_data_detect_ai(args.data, prompt_path=args.prompt_path)
    dataset = get_data(args.data, prompt_path=args.prompt_path,)
    model, tokenizer = load_model(args)

    train_dataset = Dataset(generate_data(dataset, tokenizer, args.max_length), shuffle=False)
    if "wandb" in args.report_to:
        os.environ["WANDB_PROJECT"] = args.WANDB_PROJECT
        os.environ["WANDB_API_KEY"] = args.WANDB_API_KEY
    accelerator = Accelerator()
    gradient_accumulation_steps = int(
        args.train_batch_size / args.per_device_train_batch_size / torch.cuda.device_count())
    print(gradient_accumulation_steps)
    training_args = TrainingArguments(
        bf16=args.bf16,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        adam_beta2=args.adam_beta2,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        output_dir=args.save_dir,
        report_to=args.report_to,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        deepspeed=args.deepspeed_config_path if args.deepspeed else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    model.config.use_cache = False
    trainer.train(args.resume_from_checkpoint)
    trainer.model.save_pretrained(args.save_dir)
    wandb.finish()
    trainer.save_model(args.save_dir)
