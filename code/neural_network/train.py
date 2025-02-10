import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from neuralnetwork import pileupNN
from data import load_and_split_dataset
import config

import numpy as np

device = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def training_loop(model, train_loader, criterion, optimizer):
    model.train()  # Set the model to training mode
    running_train_loss = 0.0
    batch_losses = []

    for inputs, targets in train_loader:
        # Move data to the correct device (GPU/CPU)
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())  # Store batch-wise loss
        running_train_loss += loss.item() * inputs.size(0)

    avg_train_loss = running_train_loss / len(train_loader.dataset)

    batch_loss_variance = np.var(batch_losses)
    print(f"   Batch Loss Variance: {batch_loss_variance:.6f}")  # Track loss variance
    return avg_train_loss

def validation_loop(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    running_val_loss = 0.0
    with torch.no_grad():  # Disable gradient computation during evaluation
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item() * inputs.size(0)  # Scale by batch size

    avg_val_loss = running_val_loss / len(val_loader.dataset)  # Normalize by total samples
    return avg_val_loss

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=10):
    running_train_losses = []
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        running_train_loss = training_loop(model, train_loader, criterion, optimizer)
        running_train_losses.append(running_train_loss)

        avg_val_loss = validation_loop(model, val_loader, criterion)
        val_losses.append(avg_val_loss)
        avg_train_loss = validation_loop(model, train_loader, criterion)
        train_losses.append(avg_train_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Running train Loss: {running_train_loss:.1f}, Train Loss: {avg_train_loss:.1f}, Validation Loss: {avg_val_loss:.1f}")

    weight_file = config.DATA_NEURAL_NETWORK + "model_weights.pth"
    torch.save(model.state_dict(), weight_file)
    print(f"Best model saved to model_weights.pth")# with validation loss: {best_val_loss:.4f}")

    plot_loss(train_losses, val_losses, running_train_losses)

def plot_loss(train_losses, val_losses, running_train_losses):
    plt.figure(figsize=(10/2.54, 7/2.54))
    plt.plot(np.arange(len(val_losses))+0.5, running_train_losses, label="Running training Loss")
    plt.plot(np.arange(len(val_losses))+1, train_losses, label="Training Loss")
    plt.plot(np.arange(len(val_losses))+1, val_losses, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('loss.pdf')

def main():
    train_dataset, val_dataset, test_dataset = load_and_split_dataset()
    print(f"Number of training samples: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=16)
    # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=16)

    model = pileupNN(input_size=1024, hidden_size=256, output_size=1024)
    model.to(device)

    # model.load_state_dict(torch.load(config.DATA_NEURAL_NETWORK + "model_weights.pth", map_location="cpu"))

    criterion = nn.MSELoss()  # Mean Squared Error loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=16)


if __name__ == "__main__":
    main()