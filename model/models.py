import torch
import torch.nn as nn
import torch.nn.functional as F

class SSGC(nn.Module):
    def __init__(self, n_feat, n_class, enable_bias):
        super(SSGC, self).__init__()
        self.graph_aug_linear = nn.Linear(in_features=n_feat, out_features=n_class, bias=enable_bias)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.graph_aug_linear(x)
        x = self.log_softmax(x)

        return x