import torch.nn as nn

class pileupNN(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, output_size=1024):
        super(pileupNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # Linear combination
        x = self.relu(x)  # ReLU on final output to ensure non-negative counts
        return x