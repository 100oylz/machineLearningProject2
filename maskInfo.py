import torch
import torch.nn as nn

class maskinfo(nn.Module):
    def __init__(self,in_features,out_features,dropout):
        super().__init__()
        self.dropout=nn.Dropout(p=dropout)
        self.linear=nn.Linear(in_features,out_features)

    def forward(self,x):
        out=self.linear(x)
