"""Generic dense segmentation loop with an explicit auxiliary-output contract."""
from contextlib import nullcontext
import torch
from tqdm.auto import tqdm
from .metrics import SegmentationMetrics


def train_one_epoch(model, optimizer, loader, device, criterion, scaler, desc="Train", return_components=False):
    model.train()
    total, count = 0., 0
    components = {}
    progress = tqdm(loader, desc=desc, dynamic_ncols=True, mininterval=0.5)
    for step, (images, targets, _) in enumerate(progress, 1):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast("cuda", dtype=torch.float16) if scaler.is_enabled() else nullcontext()
        with context:
            output = model(images, return_aux=True)
            if return_components:
                loss, parts = criterion(output, targets, return_components=True)
            else:
                loss = criterion(output, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss; check data and AMP settings")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if return_components:
            for key, value in parts.items():
                components[key] = components.get(key, 0.) + value.item() * images.shape[0]
        total += loss.item() * images.shape[0]
        count += images.shape[0]
        progress.set_postfix(loss=f"{total/count:.4f}", refresh=False)
    if return_components:
        return total / count, {key: value / count for key, value in components.items()}
    return total / count


@torch.inference_mode()
def evaluate(model, loader, device, criterion=None, desc="Validation", return_components=False):
    model.eval()
    metrics = None
    total, count = 0., 0
    components = {}
    progress = tqdm(loader, desc=desc, dynamic_ncols=True, mininterval=0.5)
    for images, targets, _ in progress:
        images, targets = images.to(device), targets.to(device)
        output = model(images, return_aux=True) if criterion is not None else model(images)
        logits = output["logits"] if isinstance(output, dict) else output
        if metrics is None:
            metrics = SegmentationMetrics(logits.shape[1])
        metrics.update(logits, targets)
        if criterion is not None:
            if return_components:
                loss, parts = criterion(output, targets, return_components=True)
                for key, value in parts.items():
                    components[key] = components.get(key, 0.) + value.item() * images.shape[0]
            else:
                loss = criterion(output, targets)
            total += loss.item() * images.shape[0]
        count += images.shape[0]
        if criterion is not None:
            progress.set_postfix(loss=f"{total/count:.4f}", refresh=False)
    result = metrics.compute()
    if criterion is not None:
        result["Loss"] = total / count
        if return_components:
            result.update({"val_" + key: value / count for key, value in components.items()})
    return result

