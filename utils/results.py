"""Final checkpoint evaluation and accuracy/complexity result exports."""
import csv
import json
from pathlib import Path
from torch.utils.data import DataLoader
from .dataloader_medical import PlantVOCDataset, validate_splits
from .train_and_eval import evaluate
from .benchmark import benchmark_model


def write_results(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    if result.get('benchmark') is not None:
        (output / 'benchmark.json').write_text(json.dumps(result['benchmark'], indent=2, allow_nan=False), encoding='utf-8')
    fields = ['split', 'best_epoch', 'Foreground IoU', 'Background IoU', 'Mean IoU',
              'Dice', 'Precision', 'Recall', 'Pixel Accuracy', 'Mean Accuracy',
              'Frequency Weighted IoU', 'parameters', 'params_m', 'gflops',
              'counted_gflops', 'flops_complete', 'latency_ms', 'fps', 'precision', 'gpu_name']
    with (output / 'results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for split in ('val', 'test'):
            if split not in result: continue
            row = dict(split=split, best_epoch=result['best_epoch'],
                       parameters=result['parameters'], params_m=result['params_m'])
            row.update({k: v for k, v in result[split].items() if k in fields})
            row.update({k: v for k, v in (result.get('benchmark') or {}).items() if k in fields})
            writer.writerow(row)


def export_final_results(model, checkpoint, checkpoint_path, root, output, device,
                         batch_size=2, workers=0, benchmark=True, warmup=50, iterations=200):
    splits = validate_splits(root)
    if splits != checkpoint['splits']:
        raise ValueError('Final evaluation requires the same recorded train/val/test IDs')
    parameters = sum(p.numel() for p in model.parameters())
    result = dict(checkpoint=str(checkpoint_path), best_epoch=checkpoint['epoch'] + 1,
                  selection_metric='validation Foreground IoU',
                  parameters=parameters, params_m=parameters / 1e6,
                  model_config=checkpoint['model_config'], metric_scale='0-1',
                  test_status='evaluated' if 'test' in splits else 'not provided at training time',
                  benchmark=None)
    for split in ('val', 'test'):
        if split not in splits: continue
        print(f'Final {split} evaluation using checkpoint epoch {result["best_epoch"]}...', flush=True)
        loader = DataLoader(PlantVOCDataset(root, split, augmentation=False),
                            batch_size=batch_size, num_workers=workers, shuffle=False,
                            pin_memory=device.type == 'cuda')
        result[split] = evaluate(model, loader, device)
        values = result[split]
        summary = ' | '.join(f'{key}={values[key]:.4f}' if values[key] is not None else f'{key}=N/A'
                             for key in ('Foreground IoU', 'Mean IoU', 'Dice', 'Precision', 'Recall', 'Pixel Accuracy'))
        print(f'{split}: {summary}', flush=True)
    # Accuracy results survive any later profiling failure.
    write_results(result, output)
    if benchmark:
        print('Measuring Params/GFLOPs/GPU FPS: batch=1, 512x512, FP32...', flush=True)
        try:
            report = benchmark_model(model, 512, warmup, iterations)
        except Exception as exc:
            report = dict(parameters=parameters, params_m=parameters / 1e6,
                          gflops=None, fps=None, latency_ms=None,
                          benchmark_error=f'{type(exc).__name__}: {exc}')
        report.update(checkpoint=str(checkpoint_path), best_epoch=result['best_epoch'],
                      model_config=checkpoint['model_config'])
        result['benchmark'] = report
        write_results(result, output)
    print(f'Final results saved: {Path(output) / "metrics.json"} and {Path(output) / "results.csv"}', flush=True)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result
