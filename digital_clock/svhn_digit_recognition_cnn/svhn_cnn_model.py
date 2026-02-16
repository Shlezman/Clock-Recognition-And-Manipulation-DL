import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.init as init


LEARNING_RATE = 1e-3
DECAY_STEPS = 8800
DECAY_GAMMA = 0.5

FILTER_SIZES = [5, 5, 5, 5, 5, 5, 5]
NUM_FILTERS = [32, 48, 64, 80, 96, 112, 128]
FC1_SIZE = 1024
FC2_SIZE = 1024
NUM_LABELS = 11 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def init_weights(layer, method='xavier'):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        if method == 'he':
            init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu', a=0.10)
        else:
            init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            init.constant_(layer.bias, 0.0)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, num_filters, filter_size, pooling=False, initializer='xavier'):
        super(ConvLayer, self).__init__()
        # Padding calculation to mimic TF 'SAME' with stride 1
        padding = (filter_size - 1) // 2
        
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=filter_size, stride=1, padding=padding)
        self.bn = nn.BatchNorm2d(num_filters)
        self.activation = nn.LeakyReLU(negative_slope=0.10)
        self.pooling = pooling
        
        if self.pooling:
            self.pool_layer = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            
        init_weights(self.conv, method=initializer)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        if self.pooling:
            x = self.pool_layer(x)
        return x


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, relu=False, initializer='xavier'):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.use_relu = relu
        if self.use_relu:
            self.activation = nn.LeakyReLU(negative_slope=0.10)
        init_weights(self.fc, method=initializer)

    def forward(self, x):
        x = self.fc(x)
        if self.use_relu:
            x = self.activation(x)
        return x


class SVHNModel(nn.Module):
    def __init__(self, num_channels=1):
        super(SVHNModel, self).__init__()
        
        # Block 1
        self.conv1 = ConvLayer(num_channels, NUM_FILTERS[0], FILTER_SIZES[0], pooling=False)
        self.conv2 = ConvLayer(NUM_FILTERS[0], NUM_FILTERS[1], FILTER_SIZES[1], pooling=True)
        self.drop1 = nn.Dropout(p=0.1) # p=dropout_rate (1 - keep_prob)

        # Block 2
        self.conv3 = ConvLayer(NUM_FILTERS[1], NUM_FILTERS[2], FILTER_SIZES[2], pooling=False)
        self.conv4 = ConvLayer(NUM_FILTERS[2], NUM_FILTERS[3], FILTER_SIZES[3], pooling=True)
        self.drop2 = nn.Dropout(p=0.1)

        # Block 3
        self.conv5 = ConvLayer(NUM_FILTERS[3], NUM_FILTERS[4], FILTER_SIZES[4], pooling=False)
        self.conv6 = ConvLayer(NUM_FILTERS[4], NUM_FILTERS[5], FILTER_SIZES[5], pooling=False)
        self.conv7 = ConvLayer(NUM_FILTERS[5], NUM_FILTERS[6], FILTER_SIZES[6], pooling=True)
        self.drop3 = nn.Dropout(p=0.5)

        # Flatten calc: 32x32 -> (pool) -> 16x16 -> (pool) -> 8x8 -> (pool) -> 4x4
        self.flat_features = 4 * 4 * NUM_FILTERS[6]
        
        # Fully Connected
        self.fc1 = FCLayer(self.flat_features, FC1_SIZE, relu=True)
        self.drop_fc = nn.Dropout(p=0.5)
        self.fc2 = FCLayer(FC1_SIZE, FC2_SIZE, relu=True)

        # Output Heads (5 digits)
        self.digit1 = FCLayer(FC2_SIZE, NUM_LABELS, relu=False)
        self.digit2 = FCLayer(FC2_SIZE, NUM_LABELS, relu=False)
        self.digit3 = FCLayer(FC2_SIZE, NUM_LABELS, relu=False)
        self.digit4 = FCLayer(FC2_SIZE, NUM_LABELS, relu=False)
        self.digit5 = FCLayer(FC2_SIZE, NUM_LABELS, relu=False)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop1(x)
        # Block 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.drop2(x)
        # Block 3
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.drop3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC
        x = self.fc1(x)
        x = self.drop_fc(x)
        x = self.fc2(x)
        
        return self.digit1(x), self.digit2(x), self.digit3(x), self.digit4(x), self.digit5(x)


model = SVHNModel(num_channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_STEPS, gamma=DECAY_GAMMA)
criterion = nn.CrossEntropyLoss()

