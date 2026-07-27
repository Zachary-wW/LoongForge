#!/usr/bin/env python3
"""Train MiniCPM-V-4.6 from LoongForge-exported collated batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoModelForImageTextToText


def _move_tensor(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    return value


def load_batch(path: Path, device: torch.device) -> dict:
    batch = torch.load(path, map_location="cpu")
    return {key: _move_tensor(value, device) for key, value in batch.items()}


def masked_lm_loss(logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    vocab_size = logits.shape[-1]
    flat_labels = labels.reshape(-1)
    flat_mask = loss_mask.reshape(-1).to(dtype=torch.float32)
    safe_labels = flat_labels.masked_fill(flat_labels < 0, 0)
    token_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size).float(),
        safe_labels,
        reduction="none",
    )
    loss_sum = torch.sum(token_loss * flat_mask)
    num_tokens = torch.clamp(flat_mask.sum(), min=1.0)
    return loss_sum / num_tokens, torch.stack([loss_sum.detach(), num_tokens.detach()])


def make_attention_mask(attn_mask: torch.Tensor) -> torch.Tensor:
    return (~attn_mask.bool()).to(dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--adam-eps", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--downsample-mode", default="4x")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_paths = sorted(Path(args.batch_dir).glob("batch_*.pt"))
    needed = args.max_steps * args.gradient_accumulation_steps
    if len(batch_paths) < needed:
        raise ValueError(f"Need at least {needed} exported batches, found {len(batch_paths)} in {args.batch_dir}.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "logging.jsonl"
    if log_path.exists():
        log_path.unlink()

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.downsample_mode = args.downsample_mode
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.train()
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=args.adam_eps,
        weight_decay=0.0,
    )
    for group in optimizer.param_groups:
        group["lr"] = 0.0

    def set_step_lr(step: int) -> float:
        if args.warmup_steps > 0 and step <= args.warmup_steps:
            lr = args.learning_rate * step / args.warmup_steps
        else:
            lr = args.learning_rate
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    writer = SummaryWriter(str(output_dir / "tensorboard"))
    batch_index = 0

    for step in range(1, args.max_steps + 1):
        lr = set_step_lr(step)
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        token_stats = []

        for _ in range(args.gradient_accumulation_steps):
            batch = load_batch(batch_paths[batch_index], device)
            batch_index += 1

            tokens = batch["tokens"]
            labels = batch["labels"]
            loss_mask = batch["loss_mask"]
            target_sizes = batch["image_grid_thw"][:, 1:].to(dtype=torch.int32)
            position_ids = batch["position_ids"]
            if position_ids.dim() == 3 and position_ids.shape[0] == 1:
                position_ids = position_ids.squeeze(0)

            outputs = model(
                input_ids=tokens,
                attention_mask=make_attention_mask(batch["attn_mask"]),
                position_ids=position_ids,
                pixel_values=batch["imgs"],
                target_sizes=target_sizes,
                use_cache=False,
                downsample_mode=args.downsample_mode,
            )
            micro_loss, stat = masked_lm_loss(outputs.logits, labels, loss_mask)
            (micro_loss / args.gradient_accumulation_steps).backward()
            micro_losses.append(float(micro_loss.detach().cpu()))
            token_stats.append(stat.cpu())

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_value = sum(micro_losses) / len(micro_losses)
        summed = torch.stack(token_stats).sum(dim=0)
        token_mean_loss = float((summed[0] / summed[1]).item())
        writer.add_scalar("lm loss", loss_value, step)
        writer.add_scalar("train/loss", loss_value, step)
        writer.add_scalar("train/token_mean_loss", token_mean_loss, step)
        writer.add_scalar("train/grad_norm", float(grad_norm), step)

        record = {
            "step": step,
            "loss": loss_value,
            "token_mean_loss": token_mean_loss,
            "grad_norm": float(grad_norm),
            "learning_rate": lr,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(json.dumps(record, separators=(",", ":")), flush=True)

    writer.close()


if __name__ == "__main__":
    main()
