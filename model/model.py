from torch import nn


class CNNStocksModule(nn.Module):
    OUT_CHANNELS = 200  # Number of CNN channels
    BIAS = True  # Whether to include the bias term for some of LSTM's equations

    def __init__(self, window_length: int):
        super(CNNStocksModule, self).__init__()
        self.cnn = nn.Conv1d(
            1,  # In channel size
            self.OUT_CHANNELS,
            window_length,  # kernel size
            bias=self.BIAS
        )
        if self.OUT_CHANNELS > 1:
            self.linear = nn.Linear(self.OUT_CHANNELS, 1, bias=False)

    def forward(self, x):
        out = self.cnn(x.unsqueeze(1))
        out = out.squeeze()
        if self.OUT_CHANNELS > 1:
            out = self.linear(out).squeeze()
        return out
