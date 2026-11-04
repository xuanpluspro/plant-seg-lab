"""GPU-only one-factor loss-weight sweep; plotting/collection require no torch."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DICE = (0.25, 0.5, 1., 1.5, 2.)
AUX = (0.05, 0.1, 0.2, 0.3, 0.4)


def settings():
    return list(dict.fromkeys([(d, 0.2) for d in DICE] + [(1., a) for a in AUX]))


def run_name(dice, aux):
    return f"dice_{dice:g}_aux_{aux:g}"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def best_row(history, epochs):
    if [r['epoch'] for r in history] != list(range(1, epochs + 1)):
        raise ValueError('Incomplete or unordered epoch history')
    for row in history:
        for key in ('Mean IoU', 'Dice'):
            value = row[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Invalid validation {key}: {value}')
    # max retains the first epoch in a tie, matching training selection.
    return max(history, key=lambda row: row['Mean IoU'])


def collect(output, manifest, plot=True):
    rows = []
    for d, a in settings():
        folder = output / run_name(d, a)
        if not (folder / 'completed.json').is_file():
            continue
        history = json.loads((folder / 'history.json').read_text(encoding='utf-8'))
        best = best_row(history, manifest['epochs'])
        rows.append(dict(run=folder.name, dice_weight=d, aux_weight=a,
                         best_epoch=best['epoch'], val_miou=best['Mean IoU'],
                         val_dice=best['Dice']))
    write_json(output / 'summary.json', rows)
    tmp = output / 'summary.csv.tmp'
    with tmp.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['run', 'dice_weight', 'aux_weight',
                                               'best_epoch', 'val_miou', 'val_dice'])
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(output / 'summary.csv')
    if plot and len(rows) == len(settings()):
        plot_rows(output, rows)
    return rows


def plot_rows(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.family': 'DejaVu Serif', 'font.size': 11}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
        for ax, key, fixed, title in zip(
                axes, ['dice_weight', 'aux_weight'], ['aux_weight', 'dice_weight'],
                ['(a) Dice weight (auxiliary = 0.2)', '(b) Auxiliary weight (Dice = 1)']):
            selected = sorted([r for r in rows if r[fixed] == (0.2 if fixed == 'aux_weight' else 1.)],
                              key=lambda r: r[key])
            x = [r[key] for r in selected]
            ax.plot(x, [100*r['val_miou'] for r in selected], 'o-', label='mIoU (all classes)')
            ax.plot(x, [100*r['val_dice'] for r in selected], 's-', label='Dice (foreground)')
            ax.set_xticks(x)
            ax.set_xlabel('Dice loss weight' if key == 'dice_weight' else 'Auxiliary loss weight')
            ax.set_title(title, fontsize=11)
            ax.grid(alpha=0.25, linestyle='--')
            ax.legend(fontsize=9)
        values = [100*r[k] for r in rows for k in ('val_miou', 'val_dice')]
        margin = max(1., (max(values)-min(values))*0.15)
        axes[0].set_ylim(max(0, min(values)-margin), min(100, max(values)+margin))
        axes[0].set_ylabel('Validation score (%)')
        fig.text(0.5, 0.02, 'Both metrics use the same epoch selected by validation mIoU. '
                 'Points are measured; lines are straight.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        for suffix in ('png', 'pdf', 'svg'):
            fig.savefig(output / f'loss_sensitivity.{suffix}', dpi=300)
        # Separate panels use identical y limits, without panel letters or footnotes.
        for ax, stem, title in zip(
                axes, ['dice_weight_sensitivity', 'aux_weight_sensitivity'],
                ['Dice weight (auxiliary = 0.2)', 'Auxiliary weight (Dice = 1)']):
            single, panel = plt.subplots(figsize=(5, 4))
            for line in ax.lines:
                panel.plot(line.get_xdata(), line.get_ydata(),
                           color=line.get_color(), marker=line.get_marker(),
                           linestyle=line.get_linestyle(), label=line.get_label())
            panel.set_xticks(ax.get_xticks())
            panel.set_ylim(axes[0].get_ylim())
            panel.set_xlabel(ax.get_xlabel())
            panel.set_ylabel('Validation score (%)')
            panel.set_title(title, fontsize=11)
            panel.grid(alpha=0.25, linestyle='--')
            panel.legend(fontsize=9)
            single.tight_layout()
            for suffix in ('png', 'pdf', 'svg'):
                single.savefig(output / f'{stem}.{suffix}', dpi=300)
            plt.close(single)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path')
    p.add_argument('--output', required=True)
    p.add_argument('--backbone-weights', default=str(ROOT / 'pretrained/resnet50.pth'))
    p.add_argument('--num-classes', type=int, default=1, help='Foreground count excluding background')
    p.add_argument('--mask-encoding', choices=['auto', 'index', 'binary255'], default='index')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', action='store_true', help='Resume an identical sweep, skipping completed runs')
    p.add_argument('--collect-only', action='store_true', help='Rebuild summary/figures without inference or training')
    p.add_argument('--dry-run', action='store_true', help='Print nine commands; do not import torch or create outputs')
    args = p.parse_args()
    output = Path(args.output).resolve()
    if args.collect_only:
        manifest = json.loads((output / 'sweep.json').read_text(encoding='utf-8'))
        print(f'Collected {len(collect(output, manifest))}/9 completed runs')
        return
    if not args.data_path:
        p.error('--data-path is required for training/dry-run')
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        p.error('Invalid epochs/batch-size/workers')
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error('Invalid learning rate/weight decay')
    commands = []
    for d, a in settings():
        commands.append([sys.executable, '-u', str(ROOT / 'train_medical.py'),
                         '--data-path', str(Path(args.data_path).resolve()),
                         '--output', str(output / run_name(d, a)),
                         '--backbone-weights', str(Path(args.backbone_weights).resolve()),
                         '--dice-weight', str(d), '--aux-weight', str(a),
                         '--skip-final-evaluation', '--no-benchmark',
                         '--amp' if args.amp else '--no-amp'] +
                        [part for flag, value in [('num-classes', args.num_classes),
                         ('mask-encoding', args.mask_encoding), ('epochs', args.epochs),
                         ('batch-size', args.batch_size), ('workers', args.workers),
                         ('seed', args.seed), ('lr', args.lr),
                         ('weight-decay', args.weight_decay), ('device', args.device)]
                         for part in ('--'+flag, str(value))])
    if args.dry_run:
        import shlex
        for cmd in commands:
            print(shlex.join(cmd))
        return
    import torch
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This sweep requires CUDA; CPU training is disabled')
    torch.cuda.get_device_properties(device)
    if not Path(args.backbone_weights).is_file():
        raise FileNotFoundError(args.backbone_weights)
    from utils.dataloader_medical import voc_root, validate_splits
    root = voc_root(args.data_path)
    manifest = {k: v for k, v in vars(args).items()
                if k not in ('resume', 'collect_only', 'dry_run', 'output')}
    manifest.update(data_path=str(root.resolve()),
                    backbone_weights=str(Path(args.backbone_weights).resolve()),
                    backbone_sha256=digest(args.backbone_weights), splits=validate_splits(root),
                    protocol='loss-one-factor-v1', settings=settings(),
                    code={str(f.relative_to(ROOT)): digest(f) for f in sorted(ROOT.rglob('*.py'))
                          if f.relative_to(ROOT).parts[0] in ('model', 'utils') or f.parent == ROOT})
    # Normalize tuple/list representation for comparisons after a JSON round trip.
    manifest = json.loads(json.dumps(manifest))
    if args.resume:
        previous = json.loads((output / 'sweep.json').read_text(encoding='utf-8'))
        if previous != manifest:
            raise ValueError('Resume requires identical settings, split IDs, source code and pretrained weights')
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a new output or --resume: {output}')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'sweep.json', manifest)
    for index, ((d, a), cmd) in enumerate(zip(settings(), commands), 1):
        folder = output / run_name(d, a)
        if (folder / 'completed.json').is_file():
            print(f'[{index}/9] Already completed: {folder.name}', flush=True)
            continue
        last = folder / 'last.pth'
        if last.is_file():
            pos = cmd.index('--backbone-weights')
            cmd[pos:pos+2] = ['--resume', str(last)]
        elif folder.exists() and any(folder.iterdir()):
            raise RuntimeError(f'Incomplete run without last.pth; inspect/use new output: {folder}')
        folder.mkdir(parents=True, exist_ok=True)
        print(f'[{index}/9] Training {folder.name}', flush=True)
        with (output / f'{folder.name}.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                if process.wait():
                    raise RuntimeError(f'Training failed: {folder.name}; resume with --resume')
            except BaseException:
                process.terminate()
                process.wait()
                raise
        best_row(json.loads((folder / 'history.json').read_text(encoding='utf-8')), args.epochs)
        write_json(folder / 'completed.json', {'epochs': args.epochs})
        collect(output, manifest, plot=False)
    collect(output, manifest)
    print(f'Completed 9 runs: {output}', flush=True)


if __name__ == '__main__':
    main()
