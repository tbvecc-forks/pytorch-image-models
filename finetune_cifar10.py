#!/usr/bin/env python3
""" Fine-tune a pretrained timm ViT on CIFAR-10

Compares two ways of using a pretrained backbone, both on CIFAR-10 upsampled to the model's input size:

    linear-probe   backbone frozen, only a new classifier head is trained
    finetune       every weight is trained, starting from the pretrained weights

Everything model / data / optimizer related goes through the timm API:
    timm.create_model                 pretrained weights from the Hugging Face hub, new 10 class head
    resolve_model_data_config         input size, interpolation, mean / std the weights were trained with
    create_optimizer_v2               AdamW with automatic no weight decay groups (1d params + model.no_weight_decay())
    create_scheduler_v2               per-step warmup + cosine schedule

CIFAR-10 comes from Hugging Face datasets (uoft-cs/cifar10) and is kept on the GPU as uint8. Augmentation
(random crop with padding + flip at 32x32) and the upsampling to the model resolution are done on the GPU.

Metrics are written to <output>/<mode>/metrics.json, and --plot compares all runs found in <output>.

Usage:
    python finetune_cifar10.py --mode linear-probe
    python finetune_cifar10.py --mode finetune
    python finetune_cifar10.py --plot --reference 80.5 82.2
"""
import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

import timm
from timm.data import resolve_model_data_config
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fine-tune a pretrained timm ViT on CIFAR-10')
    parser.add_argument('--model', default='vit_small_patch16_224.augreg_in21k_ft_in1k')
    parser.add_argument('--mode', choices=('linear-probe', 'finetune'), default='finetune')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=None, help='default: 1e-3 linear-probe, 1e-4 finetune')
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--warmup-epochs', type=float, default=0.5)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='output/cifar10-finetune')
    parser.add_argument('--plot', action='store_true', help='plot all runs in --output instead of training')
    parser.add_argument('--reference', type=float, nargs='*', default=(), help='from-scratch accuracies (%%) to draw as reference lines')
    return parser.parse_args()


# data

def load_cifar10(device: torch.device):
    from datasets import load_dataset

    dataset = load_dataset('uoft-cs/cifar10')

    def to_tensors(split):
        images = np.stack([np.asarray(img.convert('RGB'), dtype=np.uint8) for img in split['img']])
        images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().to(device)
        labels = torch.tensor(split['label'], dtype=torch.long, device=device)
        return images, labels

    return (*to_tensors(dataset['train']), *to_tensors(dataset['test']))


def augment(images: torch.Tensor, padding: int = 4) -> torch.Tensor:
    """ Random crop with reflect padding + random horizontal flip, per image, at native resolution. """
    batch, channels, height, width = images.shape
    device = images.device

    padded = F.pad(images, (padding,) * 4, mode='reflect')
    top = torch.randint(0, 2 * padding + 1, (batch,), device=device)
    left = torch.randint(0, 2 * padding + 1, (batch,), device=device)

    rows = (top[:, None] + torch.arange(height, device=device))[:, None, :, None]
    cols = (left[:, None] + torch.arange(width, device=device))[:, None, None, :]
    batch_index = torch.arange(batch, device=device)[:, None, None, None]
    channel_index = torch.arange(channels, device=device)[None, :, None, None]
    cropped = padded[batch_index, channel_index, rows, cols]

    flip = torch.rand(batch, device=device) < 0.5
    return torch.where(flip[:, None, None, None], cropped.flip(-1), cropped)


def preprocess(images: torch.Tensor, data_cfg: Dict, train: bool) -> torch.Tensor:
    """ uint8 32x32 -> float, (augment), upsample to the model input size, normalize with the model's mean / std. """
    x = images.float() / 255.
    if train:
        x = augment(x)

    size = data_cfg['input_size'][-2:]
    mode = 'bicubic' if data_cfg.get('interpolation') == 'bicubic' else 'bilinear'
    x = F.interpolate(x, size=size, mode=mode, align_corners=False).clamp(0., 1.)

    mean = torch.tensor(data_cfg['mean'], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(data_cfg['std'], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# model

def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = timm.create_model(args.model, pretrained=True, num_classes=10)

    if args.mode == 'linear-probe':
        # freeze everything except the new classifier (and the norm right before it, if the model has one)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True

    return model.to(device)


# evaluation

@torch.no_grad()
def evaluate(model, images, labels, data_cfg, batch_size: int = 500):
    model.eval()
    total_loss = correct = 0.

    for start in range(0, images.shape[0], batch_size):
        x = preprocess(images[start:start + batch_size], data_cfg, train=False)
        y = labels[start:start + batch_size]

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=x.is_cuda):
            logits = model(x)

        total_loss += F.cross_entropy(logits.float(), y, reduction='sum').item()
        correct += (logits.argmax(dim=-1) == y).sum().item()

    return total_loss / images.shape[0], correct / images.shape[0]


# training

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(args, device)
    data_cfg = resolve_model_data_config(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'{args.model} | mode {args.mode} | trainable {trainable:,} / {total:,} params | data cfg {data_cfg}')

    start = time.time()
    train_images, train_labels, test_images, test_labels = load_cifar10(device)
    print(f'loaded CIFAR-10 in {time.time() - start:.1f}s')

    lr = args.lr if args.lr is not None else (1e-3 if args.mode == 'linear-probe' else 1e-4)
    steps_per_epoch = train_images.shape[0] // args.batch_size

    # only trainable params reach the optimizer; timm splits them into decay / no-decay groups
    optimizer = create_optimizer_v2(model, opt='adamw', lr=lr, weight_decay=args.weight_decay)
    scheduler, _ = create_scheduler_v2(
        optimizer,
        sched='cosine',
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=lr * 1e-2,
        min_lr=0.,
        step_on_epochs=False,
        updates_per_epoch=steps_per_epoch,
    )

    run_dir = Path(args.output) / args.mode
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = dict(config=dict(vars(args), lr=lr, trainable_params=trainable, total_params=total, data_cfg=data_cfg), steps=[], epochs=[])

    test_loss, test_acc = evaluate(model, test_images, test_labels, data_cfg)
    print(f'epoch 0 (before training) | test loss {test_loss:.4f} | test acc {test_acc:.2%}')
    metrics['epochs'].append(dict(epoch=0, test_loss=test_loss, test_acc=test_acc))

    num_updates = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        permutation = torch.randperm(train_images.shape[0], device=device)
        total_loss = total_correct = seen = 0

        for step in range(steps_per_epoch):
            indices = permutation[step * args.batch_size:(step + 1) * args.batch_size]
            x = preprocess(train_images[indices], data_cfg, train=True)
            y = train_labels[indices]

            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            optimizer.step()

            num_updates += 1
            scheduler.step_update(num_updates=num_updates)

            total_loss += loss.item() * y.shape[0]
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            seen += y.shape[0]

            if num_updates % args.log_every == 0:
                metrics['steps'].append(dict(
                    step=num_updates,
                    epoch=num_updates / steps_per_epoch,
                    loss=loss.item(),
                    lr=optimizer.param_groups[0]['lr'],
                    grad_norm=grad_norm.item(),
                ))

        epoch_time = time.time() - epoch_start
        test_loss, test_acc = evaluate(model, test_images, test_labels, data_cfg)
        metrics['epochs'].append(dict(
            epoch=epoch,
            train_loss=total_loss / seen,
            train_acc=total_correct / seen,
            test_loss=test_loss,
            test_acc=test_acc,
            seconds=epoch_time,
        ))
        (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))

        print(
            f'epoch {epoch}/{args.epochs} | train loss {total_loss / seen:.4f} | train acc {total_correct / seen:.2%} | '
            f'test loss {test_loss:.4f} | test acc {test_acc:.2%} | {epoch_time:.0f}s'
        )

    print(f'final test accuracy {test_acc:.2%} | wrote {run_dir / "metrics.json"}')


# plotting

SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SERIES = {'linear-probe': '#eb6834', 'finetune': '#2a78d6'}


def plot(output: str, reference: List[float]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = {p.parent.name: json.loads(p.read_text()) for p in sorted(Path(output).glob('*/metrics.json'))}
    assert runs, f'no runs found in {output}'
    order = [name for name in ('finetune', 'linear-probe') if name in runs]

    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'font.size': 10,
        'text.color': TEXT_PRIMARY, 'axes.labelcolor': TEXT_SECONDARY, 'axes.titlecolor': TEXT_PRIMARY,
        'axes.edgecolor': GRID, 'grid.color': GRID, 'xtick.color': TEXT_SECONDARY, 'ytick.color': TEXT_SECONDARY,
        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    })

    model_name = runs[order[0]]['config']['model']
    fig, (acc_ax, loss_ax) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    fig.suptitle('Fine-tuning a pretrained ViT-S/16 on CIFAR-10', x=0.06, y=0.98, ha='left', fontsize=13, fontweight='bold')
    fig.text(
        0.06, 0.905,
        f'{model_name} · CIFAR-10 upsampled 32 → 224 · epoch 0 = before training (new random head)',
        ha='left', va='top', fontsize=9.5, color=TEXT_SECONDARY,
    )

    marker = dict(linewidth=2., marker='o', markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.2)

    # end labels, spread apart vertically when the curves finish close together
    finals = {name: 100 * runs[name]['epochs'][-1]['test_acc'] for name in order}
    label_y, previous = {}, None
    for name in sorted(order, key=lambda n: -finals[n]):
        y = finals[name] if previous is None else min(finals[name], previous - 5.)
        label_y[name] = previous = y

    for name in order:
        epochs = runs[name]['epochs']
        xs = [e['epoch'] for e in epochs]
        acc = [100 * e['test_acc'] for e in epochs]
        acc_ax.plot(xs, acc, color=SERIES[name], label=name, **marker)
        acc_ax.annotate(f'{acc[-1]:.2f}%', (xs[-1], label_y[name]), xytext=(8, 0), textcoords='offset points', va='center', fontsize=9, color=TEXT_PRIMARY)

        steps = runs[name]['steps']
        loss = [s['loss'] for s in steps]
        smoothed, avg = [], None
        for value in loss:
            avg = value if avg is None else 0.9 * avg + 0.1 * value
            smoothed.append(avg)
        loss_ax.plot([s['epoch'] for s in steps], smoothed, color=SERIES[name], linewidth=2., label=f'{name} (train)')
        loss_ax.plot(xs[1:], [e['test_loss'] for e in epochs[1:]], color=SERIES[name], linestyle=(0, (4, 3)), linewidth=1.5, label=f'{name} (test)')

    for value in reference:
        acc_ax.axhline(value, color=TEXT_SECONDARY, linewidth=1., linestyle=(0, (4, 3)))
    if reference:
        values = ' / '.join(f'{v:.1f}%' for v in reference)
        acc_ax.annotate(f'trained from scratch at 32 px, 30 epochs: {values}', (1.15, min(reference)), xytext=(0, -12),
                        textcoords='offset points', fontsize=8.5, color=TEXT_SECONDARY)

    acc_ax.set_title('test accuracy', loc='left')
    acc_ax.set_xlabel('epoch')
    acc_ax.set_ylabel('accuracy (%)')
    acc_ax.set_ylim(0, 100)
    acc_ax.set_xticks(range(0, max(e['epoch'] for e in runs[order[0]]['epochs']) + 1))
    acc_ax.legend(loc='lower right', labelcolor=TEXT_PRIMARY)

    loss_ax.set_title('loss (train: per step, EMA-smoothed, label smoothing 0.1 · test: dashed)', loc='left', fontsize=9.5)
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('cross-entropy')
    loss_ax.set_ylim(0, None)
    loss_ax.legend(loc='upper right', labelcolor=TEXT_PRIMARY, fontsize=8.5)

    for ax in (acc_ax, loss_ax):
        ax.grid(True, axis='y')

    fig.subplots_adjust(left=0.07, right=0.95, top=0.8, bottom=0.13, wspace=0.25)
    path = Path(output) / 'finetune_cifar10.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    args = parse_args()
    if args.plot:
        plot(args.output, args.reference)
    else:
        train(args)
