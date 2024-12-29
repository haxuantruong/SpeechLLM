import torch
from torch import nn


def get_connector(name, audio_enc_dim, llm_dim, k):
    if name == 'linear-pool':
        return LinearPoolConnector(audio_enc_dim, llm_dim, k)
    elif name == 'linear':
        return LinearConnector(audio_enc_dim, llm_dim, k)
    elif name == 'cnn':
        return CNNConnector(audio_enc_dim, llm_dim, k)
    else:
        raise NotImplementedError

class LinearConnector(nn.Module):
    def __init__(self, in_dim, out_dim, k):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.pool = nn.AvgPool1d(kernel_size=k, stride=k)

    def forward(self, x):
        x = self.layer(x)
        x = x.transpose(1, 2) 
        x = self.pool(x)  
        x = x.transpose(1, 2)
        return x


class LinearPoolConnector(nn.Module):
    def __init__(self, input_dim, output_dim, k):
        super(LinearPoolConnector, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU())
        self.pool = nn.AvgPool1d(kernel_size=k, stride=k)
        self.linear2 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim))

    def forward(self, x):
        # x: [B, T, d]
        x = self.linear1(x)  # x: [B, T, D]
        x = x.transpose(1, 2)  # x: [B, D, T]
        x = self.pool(x)  # x: [B, D, T']
        x = x.transpose(1, 2)  # x: [B, T', D]
        x = self.linear2(x)
        return x

class CNNConnector(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        # Chỉnh lại số lượng kênh cho các lớp Conv1d
        self.layer = nn.Sequential(
            nn.ReLU(),
            # Lớp Conv1d đầu tiên
            nn.Conv1d(in_channels, out_channels // 2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # Lớp Conv1d thứ hai, số kênh đầu vào là out_channels//2
            nn.Conv1d(out_channels // 2, out_channels, kernel_size=5, stride=k, padding=2),
            nn.ReLU(),
            # Lớp Conv1d thứ ba
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2),
        )

    def forward(self, x):
        # Đầu vào x có kích thước [B, T, in_channels]
        return self.layer(x.transpose(2, 1)).transpose(2, 1)



if __name__ == "__main__":
    cnn_connector = CNNConnector(512, 1024, k=2)  # Tham số: 512 kênh đầu vào, 1024 kênh đầu ra, stride k=2
    x = torch.randn(4, 50, 512)  # Kích thước [B, T, in_channels] = [4, 50, 512]
    z = cnn_connector(x)
    print(z.shape)  # Kết quả sẽ có kích thước [4, T', 1024] (T' phụ thuộc vào stride k)
