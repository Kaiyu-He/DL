from datetime import datetime
import json
from pathlib import Path
import time
from typing import Optional, Callable, Tuple
from functools import partial

import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets.distributed import split_dataset_by_node
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from arguments import Args
from lr_scheduler import get_wsd_lr
from optimizer import get_optimizer
from data import get_data
import safetensors.torch
from collections import OrderedDict

def print_main(*args, **kwargs):
    if accelerator.is_main_process:
        print(*args, **kwargs)


def load_train_config(args: Args):
    '''
    Load from `args.train_config` if it exists, and use the values
    when the argument is not provided.
    '''
    if args.train_config is not None and Path(args.train_config).exists():
        config = json.load(open(args.train_config, "r"))
        for k, v in config.items():
            if getattr(args, k) is None:
                setattr(args, k, v)
    else:
        print(f"WARNING: train config {args.train_config} does not exist.")
        print("It is highly recommended to provide a train config.")


def get_model(
    accelerator: Accelerator,
    args: Args,
) -> nn.Module:
    if args.model_name == 'gpt':
        from modeling import gpt
        model = gpt.get_model(accelerator, args=args)
    elif args.model_name == 'llama3':
        from modeling import llama3
        model = llama3.get_model(accelerator, args=args)
    else:
        raise ValueError

    n_non_embed_param = model.get_num_params(non_embedding=True) / 1e6
    n_param = model.get_num_params(non_embedding=False) / 1e6
    print_main("=========================================")
    print_main(f"# parameters: {n_param:.2f}M")
    print_main(f"# parameters (non-embed): {n_non_embed_param:.2f}M")
    print_main("=========================================")

    model.to(accelerator.device)
    model.to(torch.bfloat16)  # type: ignore
    return model


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data_loader: DataLoader,
    n_batches: int,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Perform evaluation on `n_batches` batches from a dataloader.
    """
    model.eval()
    losses = torch.zeros(n_batches)
    print_main("==== Evaluation ====")
    i = 0
    for i, batch in enumerate(data_loader):
        if i == n_batches:
            break
        loss, logits = model(**batch)
        losses[i] = loss.item()
    print_main("==== Evaluation Done ====")
    model.train()
    loss = losses.mean()
    return loss


def get_accelerator(args: Args) -> Accelerator:
    assert args.grad_accum_steps is not None
    assert args.run_name is not None
    assert args.project_name is not None
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        project_dir=f"./accel_logs/{args.project_name}",
        log_with=args.report_to,
    )
    hps = args.as_dict()
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d-%H%M%S")
    accelerator.init_trackers(f"{args.run_name}_{formatted_time}", config=hps)
    return accelerator


def get_dataloaders(
    args: Args, tokenizer: PreTrainedTokenizerBase, is_main_process: bool
) -> Tuple[DataLoader, Optional[DataLoader]]:
    assert args.data_name is not None
    assert args.data_path is not None
    assert args.max_len is not None
    assert args.batch_size is not None

    train_ds = get_data(
        tokenizer=tokenizer,
        data_name=args.data_name,
        data_path=args.data_path,
        max_len=args.max_len,
        is_main_process=is_main_process,
    )  # type: ignore

    # For distributed training, I think this makes sure different threads
    # fetch from different file.
    print_main(f"Number of shards: {train_ds.n_shards}")  # type: ignore
    train_ds = split_dataset_by_node(
        train_ds,  # type: ignore
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )

    train_loader = DataLoader(
        train_ds,  # type: ignore
        batch_size=args.batch_size,
        num_workers=1,
        prefetch_factor=2,
    )

    if args.validation_data_path is not None and args.validation_data_name is not None:
        val_ds = get_data(
            tokenizer=tokenizer,
            data_name=args.validation_data_name,
            data_path=args.validation_data_path,
            max_len=args.max_len,
            is_main_process=is_main_process,
        )  # type: ignore
        # val_ds = split_dataset_by_node(val_ds, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
        val_loader = DataLoader(
            val_ds,  # type: ignore
            batch_size=args.batch_size * 4,
        )
    else:
        val_loader = None

    return train_loader, val_loader


def train(
    args: Args,
    output_dir: Path,
    accelerator: Accelerator,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    get_lr: Callable[[int], float],
):
    # Training loop
    cur_step = 0  # This only increments after each model update.
    running_mfu = -1.0

    print_main("===== Start training =====")
    print_main(f"Grad accum: {args.grad_accum_steps}")
    print_main(f"Batch size: {args.batch_size}")
    print_main(f"# train iters: {args.n_train_steps}")
    print_main(f"# warmup iters: {args.n_warmup_steps}")
    print_main(f"# drop iters: {args.n_drop_steps}")
    print_main(f"Eval interval: {args.eval_interval}")
    print_main(f"Log interval: {args.log_interval}")

    # TODO: Handle resumption here
    if args.resume_model_path is not None:
        assert Path(args.resume_model_path).exists()
        # print_main(f"resume_model_path:{args.resume_model_path}")
        state_dict = safetensors.torch.load_file(args.resume_model_path)
        # print_main(f"state_dict:{state_dict}")
        print_main(f"keys:{state_dict.keys()}")
        state_dict['_orig_mod.model.lm_head.weight'] = state_dict['_orig_mod.model.input_emb.weight']

        # 去掉 _orig_mod. 前缀
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                new_key = key.replace("_orig_mod.", "")  # 去掉前缀
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        unexpected_keys, missing_keys = model.load_state_dict(new_state_dict, strict=True)
        print_main(f"unexpected_keys:{unexpected_keys}")
        print_main(f"missing_keys:{missing_keys}")
        # By default, the `lm_head` will be removed when saving with `Accelerator`.
        assert unexpected_keys == []
        assert missing_keys == []


    raw_model = model
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # for i, batch in enumerate(tqdm.tqdm(train_loader)):
    #     # print(accelerator.process_index, batch)
    #     for j, (key, value) in enumerate(batch.items()):
    #         if i >= 4340:
    #             print(f"i:{i}, key:{key},value:{value}")
    #         if value is None:
    #             print(f"Found None in batch ,key: {key},value:{value}")
    #             # 可以选择抛出异常或记录日志
    #             raise ValueError(f"Found None in batch ,key: {key},value:{value}")
    data_iter = iter(train_loader)

    # Load training states from checkpoint
    if args.resume_path is not None and args.resume_step is not None:
        # assert args.resume_model_path is None, (
        #     "You specified resume_model_path, but that will be be overridden by resume_path."
        # )
        print_main(f"Resuming from {args.resume_path} at step {args.resume_step}")
        accelerator.load_state(args.resume_path)
        cur_step = args.resume_step

        # Skip the data loader up to the resume step
        for _ in range(cur_step):
            for micro_step in range(args.grad_accum_steps):
                batch = next(data_iter, None)
                # print_main(f"batch:{batch}")
                if batch is None:
                    raise ValueError("Not enough batches in the dataloader")

    train_loss = None
    last_log_time = time.time()
    time_data = 0
    time_fwd = 0
    time_bwd = 0
    grad_norm = None

    while True:
        # TODO: Add evaluation code
        if val_loader is not None and cur_step % args.eval_interval == 0:
            val_loss = estimate_loss(
                model=model,
                data_loader=val_loader,
                n_batches=args.n_eval_batches,
            )
            accelerator.log({"loss/val": val_loss}, step=cur_step)

        # Termination condition
        if cur_step > args.n_train_steps:
            break

        # Get the LR and manually set the LR in the optimizer
        cur_lr = get_lr(cur_step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr
        for cur_micro_step in range(args.grad_accum_steps):
            start_time = time.time()
            batch = next(data_iter, None)
            # print_main(f"batch:{batch}")
            if batch is None:
                print_main("batch is None")
                continue
            time_data += time.time() - start_time
            print_main(f"time_data:{time_data}")
            with accelerator.accumulate(model):
                # Forward pass
                start_time = time.time()
                outputs = model(**batch, return_dict=True)  # type: ignore
                if isinstance(outputs, CausalLMOutputWithPast):
                    loss: Tensor = outputs.loss  # type: ignore
                else:
                    loss: Tensor = outputs[0]
                time_fwd += time.time() - start_time
                print_main(f"time_fwd:{time_fwd}")
                # Backward pass
                start_time = time.time()
                accelerator.backward(loss)
                time_bwd = time.time() - start_time
                print_main(f"time_bwd:{time_bwd}")
                # Clip the gradient
                if args.grad_clip != 0.0 and accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    )
                else:
                    grad_norm = None

                optimizer.step()
                optimizer.zero_grad()
        cur_step += 1

        # Time of this iteration
        cur_time = time.time()
        iter_time = cur_time - last_log_time
        last_log_time = cur_time

        # Logging
        if cur_step % args.log_interval == 0:
            if cur_step >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    args.grad_accum_steps
                    * args.batch_size,  # The number of sequences this process has seen.
                    iter_time,
                    train_len=args.max_len,
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            if 'loss' in locals():
                train_loss = loss.item()
                accelerator.log({"loss/train": train_loss}, step=cur_step)
                accelerator.log({"optim/lr": cur_lr}, step=cur_step)
                accelerator.log({"efficiency/mfu": running_mfu}, step=cur_step)
                accelerator.log({"efficiency/iter_time": iter_time}, step=cur_step)
                accelerator.log({"efficiency/data_time": time_data}, step=cur_step)
                accelerator.log({"efficiency/fwd_time": time_fwd}, step=cur_step)
                accelerator.log({"efficiency/bwd_time": time_bwd}, step=cur_step)

                if grad_norm is not None:
                    accelerator.log({"optim/grad_norm": grad_norm.mean()}, step=cur_step)

                time_data /= args.log_interval
                time_fwd /= args.log_interval
                time_bwd /= args.log_interval

                log_str = (
                    f"[{cur_step}/{args.n_train_steps}] loss {train_loss:.4f}"
                    f" | time {iter_time * 1000:.1f}ms"
                    f" | t_data {time_data * 1000:.1f}ms"
                    f" | t_fwd {time_fwd * 1000:.1f}ms"
                    f" | t_bwd {time_bwd * 1000:.1f}ms"
                    f" | mfu {running_mfu * 100:.1f}%"
                    f" | lr {cur_lr:.3e}"
                )
                if grad_norm is not None:
                    f" | grad_norm {grad_norm.mean():.3e}"
                print_main(log_str)

            # Reset timers
            time_data = 0
            time_fwd = 0
            time_bwd = 0

        # Checkpointing
        if accelerator.sync_gradients and cur_step % args.save_interval == 0:
            ckpt_dir = output_dir / f"ckpt_{cur_step}"
            print_main(f"Saving checkpoint to {ckpt_dir}")
            accelerator.save_state(str(ckpt_dir))

    accelerator.end_training()
    print_main("TRAINING DONE")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    args = Args().parse_args()
    load_train_config(args)
    set_seed(args.seed)
    accelerator = get_accelerator(args)

    print_main("================ args ================")
    print_main(args)
    print_main("======================================")

    # Make output dir and dump args.
    output_dir = Path(args.output_dir, args.project_name, args.run_name)
    if accelerator.is_main_process:
        output_dir.mkdir(exist_ok=True, parents=True)
        args.save(str(output_dir / "args.json"))

    # This is the actual batch size
    tokens_per_iter = (
        accelerator.num_processes
        * args.grad_accum_steps
        * args.batch_size
        * args.max_len
    )
    print_main(f"tokens per batch: {tokens_per_iter:,}")

    model: nn.Module = get_model(accelerator, args)
    tokenizer = AutoTokenizer.from_pretrained(args.tok_path)

    print_main("================ model ================")
    print_main(model)
    print_main("======================================")

    if accelerator.is_main_process:
        # save model config
        model_config = model.config
        model_config_path = output_dir / "model_config.json"
        with open(model_config_path, "w") as f:
            json.dump(model_config.to_dict(), f, indent=4, sort_keys=True)

    # TODO: Handle loading optimizer from resumed checkpoint
    optimizer = get_optimizer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        verbose=accelerator.is_main_process,
    )

    # LR scheduler, we use a function instead of a class.
    get_lr: Callable[[int], float] = partial(
        get_wsd_lr,
        lr=args.lr,
        min_lr=args.min_lr,
        n_drop_iters=args.n_drop_steps,
        n_train_iters=args.n_train_steps,
        n_warmup_iters=args.n_warmup_steps,
    )

    # Compile with PyTorch 2.0, very powerful
    if bool(args.compile):
        assert args.device != "mps", "torch.compile not supported on MPS"
        print_main("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0  # type: ignore

    train_loader, val_loader = get_dataloaders(
        args=args,
        tokenizer=tokenizer,
        is_main_process=accelerator.is_main_process,
    )

    train(
        args=args,
        output_dir=output_dir,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        get_lr=get_lr,
        train_loader=train_loader,
    )
