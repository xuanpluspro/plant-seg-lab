"""Candidate CSSP replacement: local/directional depthwise perception and spatial selection.

MPM-inspired, not the original MPM implementation or a claim of verified novelty.
"""
import torch
from torch import nn


class AdaptiveDirectionalPerception(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if channels < 8 or channels % 4:
            raise ValueError("ADPM channels must be >=8 and divisible by 4")
        hidden = channels // 4
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.branches = nn.ModuleList([
            nn.Conv2d(hidden, hidden, kernel, padding=padding, groups=hidden)
            for kernel, padding in [(3, 1), ((1, 7), (0, 3)), ((7, 1), (3, 0))]
        ])
        self.selector = nn.Sequential(nn.Conv2d(hidden * 3, hidden, 1),
                                      nn.ReLU(inplace=True), nn.Conv2d(hidden, 3, 1))
        self.project = nn.Conv2d(hidden, channels, 1)

    def forward(self, feature):
        local = self.reduce(feature)
        branches = [branch(local) for branch in self.branches]
        weights = self.selector(torch.cat(branches, dim=1)).softmax(dim=1)
        fused = sum(value * weights[:, i:i + 1] for i, value in enumerate(branches))
        return feature + self.project(fused)
