from datetime import datetime
import time

import torch
from torch import Tensor
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from tap import Tap
from accelerate import Accelerator
import evaluate
import numpy as np

from modeling.gpt import GPT, GPTConfig
from modeling.rabbit import Rabbit, RabbitConfig


accelerator = Accelerator()


class Args(Tap):
    run_name: str = 'default'
    device: str = 'cuda'

    # Training config
    lr: float = 1e-3
    weight_decay: float = 0.01
    seed: int = 0
    n_epochs: int = 8
    batch_size: int = 16
    eval_interval: int = 100
    do_eval: int = 1
    use_grad_ckpt: int = 1

    # Data config
    n_train_examples: int = 128 * 1024
    max_len: int = 512
    vocab_size: int = 1024

    # Model config
    model: str = "gpt"
    model_config: str = None
    hidden_size: int = 128
    dff: int = None
    chunk_size: int = 256
    n_layers: int = 1
    n_att_heads: int = 2
    att_head_dim: int = 64
    n_grad_steps: int = 2
    mem_dk: int = 64
    mem_dv: int = 64
    mem_width: int = 256
    mem_depth: int = 1
    mem_max_lr: float = 0.1
    use_adaptive_lr: int = 0
    mem_weight_decay: float = 0.0
    mem_beta: float = 0.9
    mem_use_norm: int = 0
    mem_use_residual: int = 0
    mem_first_layer_use_residual: int = 0


def generate_data(
    vocab_size: int,
    n_samples: int = 8192,
    max_len: int = 256,
    eos_token: int = 0,
) -> dict[str, Tensor]:
    """
    Generate copying data.

    inputs_ids = [ t1 t2 ...  tn  eos t1 ... tn-1]
    labels     = [ -100  ... -100  t1 t2 ...  tn ]

    where ki, vi are random integers in [0, vocab_size)
    """

    seq_len = (max_len - 2) // 2

    all_input_ids = []
    all_labels = []
    for i in range(n_samples):
        seq = torch.randint(16, vocab_size, (seq_len,))
        input_ids = torch.cat(
            [
                seq,
                torch.tensor([eos_token]),
                seq,
                torch.tensor([eos_token]),
            ],
            dim=0,
        )
        label_ids = input_ids.clone()
        label_ids[: seq_len + 1] = -100

        # # Shift labels
        # input_ids = input_ids[:-1]
        # label_ids = label_ids[1:]

        all_input_ids.append(input_ids)
        all_labels.append(label_ids)

    return {
        "input_ids": torch.stack(all_input_ids),
        "labels": torch.stack(all_labels),
    }


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: dict[str, Tensor]):
        self.data = data

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.data["input_ids"][idx],
            "labels": self.data["labels"][idx],
        }


def get_model(args: Args):
    if args.model == "gpt":
        from modeling.gpt import GPT, GPTConfig
        model_config = GPTConfig(
            vocab_size=args.vocab_size,
            dim=args.hidden_size,
            n_heads=args.n_att_heads,
            n_layers=args.n_layers,
            head_dim=args.att_head_dim,
        )
        model = GPT(config=model_config).to(args.device)
    elif args.model == "rabbit":
        from modeling.rabbit import Rabbit, RabbitConfig
        if args.model_config is not None:
            model_config = RabbitConfig.from_json_file(args.model_config)
        else:
            model_config = RabbitConfig(
                max_len=args.max_len,
                vocab_size=args.vocab_size,
                hidden_size=args.hidden_size,
                dff=args.dff,
                n_att_heads=args.n_att_heads,
                att_head_dim=args.att_head_dim,
                n_layers=args.n_layers,
                n_grad_steps=args.n_grad_steps,
                mem_dk=args.mem_dk,
                mem_dv=args.mem_dv,
                mem_width=args.mem_width,
                mem_depth=args.mem_depth,
                mem_max_lr=args.mem_max_lr,
                use_adaptive_lr=args.use_adaptive_lr,
                mem_weight_decay=args.mem_weight_decay,
                mem_beta=args.mem_beta,
                mem_use_norm=bool(args.mem_use_norm),
                mem_use_residual=bool(args.mem_use_residual),
                mem_first_layer_use_residual=bool(args.mem_first_layer_use_residual),
                chunk_size=args.chunk_size,
                use_grad_ckpt=args.use_grad_ckpt,
            )
        model = Rabbit(config=model_config).to(args.device)
    elif args.model == "gdn":
        from modeling.gated_deltanet.model import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
        model_config = GatedDeltaNetConfig(
            hidden_size=args.hidden_size,
            vocab_size=args.vocab_size,
            head_dim=args.att_head_dim,
            num_heads=args.n_att_heads,
            max_position_embeddings=args.max_len,
            num_hidden_layers=args.n_layers,
            tie_word_embeddings=True,
            pad_token_id=0,
            bos_token_id=0,
            eos_token_id=0,
        )
        model = GatedDeltaNetForCausalLM(config=model_config)
        model.to(dtype=torch.bfloat16)  # type: ignore
    elif args.model == 'mamba2':
        from modeling.mamba2.model import Mamba2Config, Mamba2ForCausalLM
        model_config = Mamba2Config(
            num_heads=args.n_att_heads,
            head_dim=args.att_head_dim,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            state_size=args.att_head_dim * 2,
            num_hidden_layers=args.n_layers,
            pad_token_id=0,
            bos_token_id=0,
            eos_token_id=0,
            expand=2,
        )
        model: Mamba2ForCausalLM = Mamba2ForCausalLM(config=model_config)
        if args.use_grad_ckpt:
            model.gradient_checkpointing_enable()
    else:
        raise ValueError(f"Unknown model name: {args.model}")

    accelerator.print(model)
    accelerator.print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    accelerator.print(f"Initial GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6} MB")
    accelerator.print(f"Max GPU Memory Allocated: {torch.cuda.max_memory_allocated() / 1e6} MB")
    return model


class StopCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if len(state.log_history) > 0 and 'log' in state.log_history[-1]:
            last_loss = state.log_history[-1]['loss']
            if last_loss < 0.05:
                control.should_training_stop = True
        return control


import time
class MemoryTrackerCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None  # 记录步长开始时间

    def on_step_begin(self, args, state, control, **kwargs):
        """ 训练步长开始时记录时间 """
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        """ 训练步长结束时计算运行时间，并记录显存使用情况 """
        if self.start_time is not None:
            step_time = time.time() - self.start_time  # 计算步长耗时
        else:
            step_time = None

        # 当前显存占用
        current_memory = torch.cuda.memory_allocated() / 1e6  # MB
        # 训练过程中的最大显存使用量
        max_memory = torch.cuda.max_memory_allocated() / 1e6  # MB

        accelerator.print(
            f"Step {state.global_step}: Time = {step_time:.3f} s, "
            f"GPU Memory Allocated = {current_memory:.2f} MB, "
            f"Max Memory Used = {max_memory:.2f} MB"
        )


def compute_metrics(eval_pred):
    logits, labels = eval_pred  # (1024, 512, 1024), (1024, 512)
    logits = logits[:, :-1]
    labels = labels[:, 1:]
    preds = np.argmax(logits, axis=-1)  # token predictions
    # print('preds', preds.shape, type(preds), preds)  # (1024, 1024)
    # time.sleep(1)
    # print('labels', labels.shape, type(labels), labels)  # (1024, 512)
    mask = labels != -100
    preds, labels = preds[mask], labels[mask]
    accelerator.print(f"GPU Memory During Eval: {torch.cuda.memory_allocated() / 1e6} MB")
    return {"accuracy": (preds == labels).mean()}


def main():
    args = Args().parse_args()
    accelerator.print("============ args ============")
    accelerator.print(args)
    accelerator.print("==============================")

    data = generate_data(
        vocab_size=args.vocab_size,
        n_samples=args.n_train_examples,
        max_len=args.max_len,
    )
    eval_data = generate_data(
        vocab_size=args.vocab_size,
        n_samples=1024,
        max_len=args.max_len,
    )

    train_dataset = Dataset(data)
    eval_dataset = Dataset(eval_data)

    model = get_model(args)

    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M%S')
    run_name = f'{args.model}_{args.run_name}_{formatted_time}'

    train_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.n_epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=1,
        logging_first_step=True,
        do_train=True,
        do_eval=bool(args.do_eval),
        eval_on_start=True,
        eval_steps=args.eval_interval,
        eval_strategy='steps',
        run_name=run_name,
        logging_dir=f"./results/tensorboard/{run_name}",
        save_safetensors=False,
        weight_decay=args.weight_decay,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[StopCallback(),MemoryTrackerCallback()],
        compute_metrics=compute_metrics,
    )
    accelerator.print("===== Start training =====")
    start_train = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    print(f"Before train - Memory allocated: {start_train:.2f} MB")
    trainer.train()
    # torch.cuda.empty_cache()
    accelerator.print(eval_dataset)
    torch.cuda.reset_peak_memory_stats()  # 先重置峰值统计
    start_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    start_reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
    start_max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    print(
        f"Before eval - Allocated: {start_mem:.2f} MB, Reserved: {start_reserved:.2f} MB, Max: {start_max_mem:.2f} MB")

    result = trainer.evaluate(eval_dataset)

    end_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    end_reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
    end_max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    print(f"After eval - Allocated: {end_mem:.2f} MB, Reserved: {end_reserved:.2f} MB, Max: {end_max_mem:.2f} MB")
    print(f"Peak memory during eval: {end_max_mem - start_max_mem:.2f} MB")
    print(f"result:{result}")

if __name__ == "__main__":
    main()