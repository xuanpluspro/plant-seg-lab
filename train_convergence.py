"""Train one 80-epoch convergence run on CUDA, using the normal training pipeline."""
from train_medical import parse_args, train
import torch


if __name__ == '__main__':
    args = parse_args(default_epochs=80, default_device='cuda')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Convergence training requires CUDA; CPU fallback is disabled')
    torch.cuda.get_device_properties(device)
    # Epoch validation/checkpoints remain enabled; no automatic test-set use.
    args.skip_final_evaluation = True
    args.convergence = True
    train(args)
