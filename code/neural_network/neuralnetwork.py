import torch.nn as nn
import config

class pileupNN(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, output_size=config.DIM_OUTPUT_PARAMETERS * 2):
        """Output dimensions: First half are the model parameters, the last half are the log variance."""
        super(pileupNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

        self.activation = nn.Softplus()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)  # Linear combination
        return x