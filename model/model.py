from torch import nn
import torch


class CNNStocksModule(nn.Module):
    OUT_CHANNELS = 30  # Number of CNN channels
    KERNEL_SIZE = 10  # Size of CNN kernel

    def __init__(self, window_length: int):
        super(CNNStocksModule, self).__init__()

        assert window_length >= self.KERNEL_SIZE
        self.cnn = nn.Conv1d(
            1,  # In channel size
            self.OUT_CHANNELS,
            self.KERNEL_SIZE
        )

        num_scores = window_length - self.KERNEL_SIZE + 1

        # MaxPool kernel size is set such that we only output one value for each row/channel
        self.pool = nn.MaxPool1d(num_scores)

        self.linear = nn.Linear(self.OUT_CHANNELS, 1, bias=True)

    def forward(self, x):
        out = self.cnn(x.unsqueeze(1))
        out = self.pool(out).squeeze()
        out = torch.softmax(out, dim=1)
        out = self.linear(out).squeeze()
        return out
