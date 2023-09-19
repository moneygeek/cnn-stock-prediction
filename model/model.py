from torch import nn
import torch


class CNNStocksModule(nn.Module):
    OUT_CHANNELS = 30  # Number of CNN channels
    KERNEL_SIZE = 10  # Size of CNN kernel
    BIAS = True  # Whether to include the bias term for some of CNN's equations

    def __init__(self, window_length: int):
        super(CNNStocksModule, self).__init__()
        self.cnn = nn.Conv1d(
            1,  # In channel size
            self.OUT_CHANNELS,
            self.KERNEL_SIZE,
            bias=self.BIAS
        )

        # MaxPool kernel size is set such that we only output one value for each row/channel
        size_horizontal = window_length - self.KERNEL_SIZE + 1
        self.pool = nn.MaxPool1d(size_horizontal)

        self.linear = nn.Linear(self.OUT_CHANNELS, 1, bias=True)

    def forward(self, x):
        out = self.cnn(x.unsqueeze(1))
        out = self.pool(out)
        out = torch.softmax(out, dim=1)
        out = torch.flatten(out, start_dim=1)
        out = self.linear(out).squeeze()
        return out
