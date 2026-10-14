# Backbone pretrained weights

Default file: pretrained/resnet50.pth (official torchvision ResNet-50,
ImageNet-1K V2). Run `python download_pretrained.py` from the project.
The script verifies the SHA256 prefix published in torchvision's download URL,
reuses a verified file, and does not overwrite a mismatched existing file.
Binary weights and unfinished downloads are ignored by Git.

This file initializes only the ResNet-50 backbone. ADPM, dynamic fusion and
segmentation heads are trained on your plant masks; it is not a full
segmentation checkpoint. Other compatible local torchvision RGB ResNet-50
weights can be selected explicitly with --backbone-weights /path/to/file.pth.
