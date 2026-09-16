import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self , input_size = 784 , hidden_size = [256 , 128] , num_classes = 10):
        super(SimpleNet , self).__init__()

        # Create a list of layer size  [784] + [256, 128] + [10]  →  [784, 256, 128, 10]

        layer_sizes = [input_size] + hidden_size + [num_classes]

        # Create the layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            # (784 , 256) -> (256 , 128) -> (128 , 10)
            self.layers.append(nn.Linear(layer_sizes[i] , layer_sizes[i + 1]))

    def forward(self , x):
        # Flatten the input (batch_size , 1 , 28 , 28) -> (batch_size , 784)
        x = torch.flatten(x , 1)

        # Forward through all layers except the last one with ReLU activation
        for layer in self.layers[ : -1]:
            x = F.relu(layer(x))

        #Last layer without activation
        x = self.layers[-1](x)

        return F.log_softmax(x , dim = 1)