import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, hid_1, hid_2, hid_3, hid_4, hid_5, hid_6, x, y, drop_1, drop_2):
        super().__init__()

        # Layer convoluzionali
        self.conv1 = nn.Conv2d(3, hid_1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hid_1)
        self.conv2 = nn.Conv2d(hid_1, hid_2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(hid_2)

        # Secondo blocco di convoluzione
        self.conv3 = nn.Conv2d(hid_2, hid_3, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(hid_3)
        self.conv4 = nn.Conv2d(hid_3, hid_4, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(hid_4)

        # Terzo blocco di convoluzione
        self.conv5 = nn.Conv2d(hid_4, hid_5, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(hid_5)
        self.conv6 = nn.Conv2d(hid_5, hid_6, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(hid_6)

        # Fully connected layers
        self.fc1 = nn.Linear(hid_6 * x * y, 512)  # Dimensioni finali dopo convoluzioni
        self.bn7 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 8)  # Classe finale

        # MaxPooling, Dropout2d, Dropout Fully Connected
        self.pool = nn.MaxPool2d(2,2)
        self.dropout2d = nn.Dropout2d(drop_1)
        self.dropout = nn.Dropout(drop_2)

    def forward(self, x):
        # Primo blocco convoluzionale
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Secondo blocco convoluzionale
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))

        # Terzo blocco convoluzionale
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)

        # Dropout e Flatten
        x = self.dropout2d(x)
        x = torch.flatten(x, 1)  # Flatten

        # Fully connected
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)  # Output layer

        return x