"""Export empirical convergence curves from history; no torch or model needed."""
import argparse
import csv
import json
import math
from pathlib import Path

COMPONENTS = ('ce', 'dice', 'aux', 'weighted_ce', 'weighted_dice', 'weighted_aux')


def convergence_rows(history):
    rows = []
    previous = 0
    for entry in history:
        epoch = entry['epoch']
        if not isinstance(epoch, int) or epoch <= previous:
            raise ValueError('Epochs must be strictly increasing positive integers')
        previous = epoch
        row = dict(epoch=epoch, train_total=entry['train_loss'],
                   val_total=entry['Loss'], val_miou=entry['Mean IoU'],
                   val_dice=entry['Dice'], lr=entry.get('lr'))
        for split in ('train', 'val'):
            for component in COMPONENTS:
                key = split + '_' + component
                row[key] = entry.get(key)
        for key, value in row.items():
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f'Invalid {key} at epoch {epoch}')
        if not 0 <= row['val_miou'] <= 1 or not 0 <= row['val_dice'] <= 1:
            raise ValueError('Validation scores must be in [0,1]')
        rows.append(row)
    if not rows:
        raise ValueError('No history to plot')
    return rows


def export_convergence(history, output):
    rows = convergence_rows(history)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    temp = output / 'convergence.csv.tmp'
    with temp.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(output / 'convergence.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [row['epoch'] for row in rows]

    def save(fig, stem):
        fig.tight_layout()
        for extension in ('png', 'svg', 'pdf'):
            fig.savefig(output / f'{stem}.{extension}', dpi=300)
        plt.close(fig)

    def loss_panel(ax):
        ax.plot(epochs, [r['train_total'] for r in rows], label='Train total loss')
        ax.plot(epochs, [r['val_total'] for r in rows], label='Validation total loss')
        ax.set_ylabel('Loss')
        ax.set_title('Total loss')

    def metrics_panel(ax):
        ax.plot(epochs, [100*r['val_miou'] for r in rows], label='mIoU (all classes)')
        ax.plot(epochs, [100*r['val_dice'] for r in rows], label='Dice (foreground)')
        ax.set_ylabel('Validation score (%)')
        ax.set_title('Validation performance')

    def decorate(ax):
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.25, linestyle='--')
        ax.legend(fontsize=9)

    with plt.rc_context({'font.family': 'DejaVu Serif', 'font.size': 11}):
        for draw, stem in ((loss_panel, 'convergence_loss'),
                           (metrics_panel, 'convergence_metrics')):
            fig, ax = plt.subplots(figsize=(6, 4))
            draw(ax)
            decorate(ax)
            save(fig, stem)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        loss_panel(axes[0])
        metrics_panel(axes[1])
        for ax in axes:
            decorate(ax)
        save(fig, 'convergence_combined')
        # Missing historical components remain blank/NaN, never reconstructed.
        for split in ('train', 'val'):
            if not any(r[split+'_weighted_ce'] is not None for r in rows):
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            for key, label in (('weighted_ce', 'CE'), ('weighted_dice', 'Weighted Dice loss'),
                               ('weighted_aux', 'Weighted auxiliary BCE')):
                ax.plot(epochs, [r[split+'_'+key] if r[split+'_'+key] is not None
                                 else float('nan') for r in rows], label=label)
            ax.set_title('Training loss components' if split == 'train' else 'Validation loss components')
            ax.set_ylabel('Weighted loss')
            decorate(ax)
            save(fig, f'convergence_components_{split}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Run directory containing history.json')
    args = parser.parse_args()
    run = Path(args.run)
    export_convergence(json.loads((run/'history.json').read_text(encoding='utf-8')), run)
