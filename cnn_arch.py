import torch
import torch.nn as nn
import torch.nn.functional as F


# the DCE network for Pilot_num 128
class DCE_N(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_features = 32
        self.kernel_size = 3
        self.pad = 1
        conv_layers = []
        #1st layer
        conv_layers.append(nn.Conv2d(in_channels=2, out_channels=self.num_features, kernel_size=self.kernel_size, stride=1,
                                padding=self.pad,
                                bias=False))
        conv_layers.append(nn.BatchNorm2d(self.num_features))
        conv_layers.append(nn.ReLU(inplace=True))

        #2nd layer
        conv_layers.append(
            nn.Conv2d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=self.kernel_size, stride=1,
                        padding=self.pad,
                        bias=False))
        conv_layers.append(nn.BatchNorm2d(self.num_features))
        conv_layers.append(nn.ReLU(inplace=True))
        
        #3rd Layer
        conv_layers.append(
            nn.Conv2d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=self.kernel_size, stride=1,
                        padding=self.pad,
                        bias=False))
        conv_layers.append(nn.BatchNorm2d(self.num_features))
        conv_layers.append(nn.ReLU(inplace=True))
        

        self.cnn = nn.Sequential(*conv_layers)

        #Linear layer
        self.FC = nn.Linear(self.num_features * 16 * 8, 64 * 16 * 2)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], self.num_features * 16 * 8)
        x = self.FC(x)
        return x

# Scenario classifier for Pilot_num 128
class SC_N(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.FC = nn.Linear(32 * 4 * 2, 3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Apply convolution, batch normalization, ReLU, and max pooling sequentially
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        # Flatten the output
        x = x.view(x.shape[0], -1)
        # Pass through the linear layer
        x = self.FC(x)
    
        return F.log_softmax(x, dim=1)


# the feature extractor for Pilot_num 128
class Conv_N(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = 32
        self.kernel_size = 3
        self.padding = 1
        conv_layers = []
        # the first layer
        conv_layers.append(
            nn.Conv2d(in_channels=2, out_channels=self.features, kernel_size=self.kernel_size, padding=self.padding,
                      bias=False))
        conv_layers.append(nn.BatchNorm2d(self.features))
        conv_layers.append(nn.ReLU(inplace=True))

        conv_layers.append(nn.Conv2d(in_channels=self.features, out_channels=self.features, kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=False))
        conv_layers.append(nn.BatchNorm2d(self.features))
        conv_layers.append(nn.ReLU(inplace=True))
        
        conv_layers.append(nn.Conv2d(in_channels=self.features, out_channels=self.features, kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=False))
        conv_layers.append(nn.BatchNorm2d(self.features))
        conv_layers.append(nn.ReLU(inplace=True))

        self.cnn = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], self.features * 16 * 8)

        return x

# the feature mapper for Pilot_num 128
class FC_N(nn.Module):
    def __init__(self):
        super().__init__()
        self.FC = nn.Linear(32 * 16 * 8, 64 * 16 * 2)

    def forward(self, x):
        x = self.FC(x)
        return x

class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, x_hat, x):
        # Calc the squared error
        mse = torch.sum((x_hat - x).pow(2))
        # Calc the power of the signal
        power = torch.sum(x.pow(2))
        # Calc the NMSE by normalizing the MSE by the power
        nmse = mse / power
        return nmse