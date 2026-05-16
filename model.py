import torch
import torch.nn as nn


class simplecnn(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class inceptionblock(nn.Module):
    def __init__(self, in_channels, out1, out3, out5, out_pool):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out1, kernel_size=1),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out3, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, out5, kernel_size=5, padding=2),
            nn.ReLU()
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch3 = self.branch3(x)
        branch5 = self.branch5(x)
        branch_pool = self.branch_pool(x)

        output = torch.cat(
            [branch1, branch3, branch5, branch_pool],
            dim=1
        )

        return output


class googlenetlike(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            inceptionblock(32, 16, 16, 16, 16),
            nn.MaxPool2d(2),

            inceptionblock(64, 32, 32, 32, 32),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x