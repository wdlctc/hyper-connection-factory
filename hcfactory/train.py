"""Training template shared by every connection variant.

Single device:
    python -m hcfactory.train --config configs/tiny.yaml connection=mhc
Multi-GPU:
    torchrun --nproc_per_node=8 -m hcfactory.train --config configs/gpt124m.yaml connection=hc

Any config key can be overridden as ``key=value`` (``connection_kwargs.n=4`` for
nested keys). Every run writes ``<out_dir>/<run_name>/{config.json,log.jsonl,summary.json}``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from dataclasses import asdict

import torch
import torch.distributed as dist
import yaml

from .data import TokenLoader
from .model import GPT, ModelConfig

DEFAULTS = dict(
    # model
    n_layer=8, n_head=8, d_model=512, mlp_ratio=8 / 3, vocab_size=50304,
    connection="prenorm", connection_kwargs={},
    # data
    data_dir="data/fineweb-edu", seq_len=1024, batch_size=16, grad_accum=1,
    # optimisation
    max_steps=2000, lr=3e-3, min_lr_ratio=0.1, warmup_steps=100, schedule="cosine",
    cooldown_frac=0.2, weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0,
    connection_lr_mult=1.0,
    # eval / logging
    eval_every=200, eval_batches=20, log_every=10,
    out_dir="runs", run_name=None, seed=0, compile=False, dtype="auto",
    wandb_project=None,
)


def parse_value(v: str):
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def load_config(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*", help="key=value")
    args = ap.parse_args(argv)
    cfg = json.loads(json.dumps(DEFAULTS))
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f) or {})
    for ov in args.overrides:
        k, v = ov.split("=", 1)
        d = cfg
        *parents, leaf = k.split(".")
        for p in parents:
            d = d.setdefault(p, {})
        d[leaf] = parse_value(v)
    if cfg["run_name"] is None:
        kw = "".join(f"_{k}{v}" for k, v in sorted(cfg["connection_kwargs"].items()))
        cfg["run_name"] = f"{cfg['connection']}{kw}_s{cfg['seed']}"
    return cfg


def lr_at(step: int, cfg) -> float:
    base, warm, total = cfg["lr"], cfg["warmup_steps"], cfg["max_steps"]
    lo = base * cfg["min_lr_ratio"]
    if step < warm:
        return base * (step + 1) / warm
    if cfg["schedule"] == "wsd":  # warmup-stable-decay
        start = int(total * (1 - cfg["cooldown_frac"]))
        if step < start:
            return base
        return lo + (base - lo) * (1 - (step - start) / max(1, total - start))
    t = (step - warm) / max(1, total - warm)
    return lo + 0.5 * (base - lo) * (1 + math.cos(math.pi * min(t, 1.0)))


def main(argv=None):
    cfg = load_config(argv)

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local)
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    master = rank == 0
    torch.manual_seed(cfg["seed"])

    dtype = cfg["dtype"]
    if dtype == "auto":
        dtype = "bfloat16" if device.type == "cuda" else "float32"
    amp = (
        torch.autocast(device.type, dtype=getattr(torch, dtype))
        if dtype != "float32"
        else contextlib.nullcontext()
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    mcfg = ModelConfig(
        vocab_size=cfg["vocab_size"], n_layer=cfg["n_layer"], n_head=cfg["n_head"],
        d_model=cfg["d_model"], mlp_ratio=cfg["mlp_ratio"], max_seq_len=cfg["seq_len"],
        connection=cfg["connection"], connection_kwargs=cfg["connection_kwargs"],
    )
    model = GPT(mcfg).to(device)
    n_params = model.num_params()
    n_conn = model.connection_params()

    # Connection parameters (gates, routing queries, mixing matrices) get no
    # weight decay: decaying them pulls every variant back towards "no mixing".
    decay, no_decay, conn = [], [], []
    for name, p in model.named_parameters():
        if name.startswith("connection."):
            conn.append(p)
        elif p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        dict(params=decay, weight_decay=cfg["weight_decay"], lr_mult=1.0),
        dict(params=no_decay, weight_decay=0.0, lr_mult=1.0),
        dict(params=conn, weight_decay=0.0, lr_mult=cfg["connection_lr_mult"]),
    ]
    opt = torch.optim.AdamW(
        groups, lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]),
        fused=device.type == "cuda",
    )

    fwd = torch.compile(model) if cfg["compile"] else model
    if ddp:
        fwd = torch.nn.parallel.DistributedDataParallel(fwd, device_ids=[device.index])

    train = TokenLoader(os.path.join(cfg["data_dir"], "train.bin"), cfg["batch_size"],
                        cfg["seq_len"], seed=cfg["seed"] * 1000 + rank, device=device)
    val = TokenLoader(os.path.join(cfg["data_dir"], "val.bin"), cfg["batch_size"],
                      cfg["seq_len"], device=device)
    val_batches = val.fixed_batches(cfg["eval_batches"])
    tokens_per_step = cfg["batch_size"] * cfg["seq_len"] * cfg["grad_accum"] * world

    run_dir = os.path.join(cfg["out_dir"], cfg["run_name"])
    log_f = None
    if master:
        os.makedirs(run_dir, exist_ok=True)
        meta = dict(cfg=cfg, model=asdict(mcfg), params_non_embedding=n_params,
                    connection_params=n_conn, device=str(device), world_size=world,
                    tokens_per_step=tokens_per_step)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(meta, f, indent=2)
        log_f = open(os.path.join(run_dir, "log.jsonl"), "w")
        print(f"[{cfg['run_name']}] params(non-emb)={n_params / 1e6:.2f}M "
              f"connection={n_conn / 1e3:.1f}K tokens/step={tokens_per_step} device={device}")
        if cfg["wandb_project"]:
            import wandb
            wandb.init(project=cfg["wandb_project"], name=cfg["run_name"], config=meta)

    def log(rec):
        if not master:
            return
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if cfg["wandb_project"]:
            import wandb
            wandb.log(rec, step=rec["step"])

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot = torch.zeros((), device=device)
        for x, y in val_batches:
            with amp:
                _, loss = model(x, y)
            tot += loss.float()
        model.train()
        tot /= len(val_batches)
        if ddp:
            dist.all_reduce(tot, op=dist.ReduceOp.AVG)
        return tot.item()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    best_val, diverged = float("inf"), False
    t_train, tok_seen = 0.0, 0
    for step in range(cfg["max_steps"] + 1):
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            vl = evaluate()
            best_val = min(best_val, vl)
            log(dict(step=step, tokens=tok_seen, val_loss=vl))
            if master:
                print(f"step {step:6d} | val {vl:.4f}")
            if not math.isfinite(vl):
                diverged = True
                break
        if step == cfg["max_steps"]:
            break

        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_mult"]
        sync()
        t0 = time.perf_counter()
        loss_acc = 0.0
        for micro in range(cfg["grad_accum"]):
            x, y = train.next()
            ctx = (fwd.no_sync() if ddp and micro < cfg["grad_accum"] - 1
                   else contextlib.nullcontext())
            with ctx, amp:
                _, loss = fwd(x, y)
                loss = loss / cfg["grad_accum"]
            loss.backward()
            loss_acc += loss.detach()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        dt = time.perf_counter() - t0
        if step > 5:  # skip compile / warmup steps in the throughput figure
            t_train += dt
        tok_seen += tokens_per_step

        if step % cfg["log_every"] == 0:
            lv = float(loss_acc)
            rec = dict(step=step, tokens=tok_seen, loss=lv, lr=lr, grad_norm=float(gnorm),
                       tok_per_s=tokens_per_step / dt)
            if device.type == "cuda":
                rec["mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            log(rec)
            if master and step % (cfg["log_every"] * 10) == 0:
                print(f"step {step:6d} | loss {lv:.4f} | lr {lr:.2e} | gnorm {float(gnorm):.2f}"
                      f" | {tokens_per_step / dt / 1e3:.1f}K tok/s")
            if not math.isfinite(lv):
                diverged = True
                break

    if master:
        steps_timed = max(1, min(step, cfg["max_steps"]) - 6)
        summary = dict(
            run_name=cfg["run_name"], connection=cfg["connection"],
            connection_kwargs=cfg["connection_kwargs"], seed=cfg["seed"],
            params_non_embedding=n_params, connection_params=n_conn,
            final_val_loss=vl, best_val_loss=best_val, tokens=tok_seen, diverged=diverged,
            tok_per_s=tokens_per_step * steps_timed / max(t_train, 1e-9),
            peak_mem_gb=(torch.cuda.max_memory_allocated() / 1e9
                         if device.type == "cuda" else None),
            device=str(device), world_size=world,
        )
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        log_f.close()
        print(json.dumps(summary))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
