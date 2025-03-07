import matplotlib.pyplot as plt
import csv
import numpy as np
from dataclasses import dataclass,fields

import torch
from torch.utils.data import DataLoader

from neuralnetwork import pileupNN
from data import load_and_split_dataset
import config

device = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class EpochMetadata:
    running_train_loss: float
    mean_total_grad_norm: float
    batch_losses: list[float]
    total_grad_norms: list[float]

@dataclass
class TrainMetadata:
    training_losses: list[float]
    validation_losses: list[float]
    running_train_losses: list[float]
    mean_total_grad_norms: list[float]

def _get_grad_norm(model: torch.nn.Module) -> float:
    grad_norms = []
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm(2).item()  # L2 norm of the gradients
            grad_norms.append(grad_norm)
            total_norm += grad_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def training_loop(model, train_loader, criterion, optimizer):
    model.train()  # Set the model to training mode
    running_train_loss = 0.0
    batch_losses = []
    total_grad_norms = []

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move data to the correct device (GPU/CPU)
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()

        total_norm = _get_grad_norm(model)
        total_grad_norms.append(total_norm)

        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        batch_losses.append(loss.item())  # Store batch-wise loss
        running_train_loss += loss.item() * inputs.size(0)

    running_train_loss = running_train_loss / len(train_loader.dataset)
    mean_total_grad_norm = np.mean(total_grad_norms)

    metadata = EpochMetadata(running_train_loss = running_train_loss, mean_total_grad_norm = mean_total_grad_norm,
                             batch_losses = batch_losses, total_grad_norms = total_grad_norms)

    return metadata

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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_metadata = TrainMetadata(*[list() for dummy in fields(TrainMetadata)])

    for epoch in range(num_epochs):
        epoch_metadata = training_loop(model, train_loader, criterion, optimizer)
        train_metadata.running_train_losses.append(epoch_metadata.running_train_loss)
        train_metadata.mean_total_grad_norms.append(epoch_metadata.mean_total_grad_norm)

        avg_val_loss = validation_loop(model, val_loader, criterion)
        train_metadata.validation_losses.append(avg_val_loss)
        avg_train_loss = validation_loop(model, train_loader, criterion)
        train_metadata.training_losses.append(avg_train_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.1f}, Validation Loss: {avg_val_loss:.1f}")

    weight_file = config.DATA_NEURAL_NETWORK + "model_weights.pth"
    torch.save(model.state_dict(), weight_file)
    print(f"Best model saved to model_weights.pth")

    write_metadata_file(train_metadata)
    plot_loss(train_metadata, label = type(criterion).__name__)

def write_metadata_file(metadata, csv_filename = "loss.csv"):
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = ["Epoch"] + [field.name for field in fields(metadata)]
        writer.writerow(header)
        all_fields = [getattr(metadata, field.name) for field in fields(metadata)]
        for epoch, values in enumerate(zip(*all_fields), start=1):
            writer.writerow([epoch] + list(values))

def plot_loss(metadata, label = "Loss"):
    fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(12/2.54, 12/2.54))
    # Note that running training loss (not plotted here) should start half an epoch
    # earlier (at 0.5) because it is calculated during training while the gradients
    # are still updated.
    axes[0].plot(np.arange(len(metadata.training_losses))+1, metadata.training_losses, label="Training Loss")
    axes[0].plot(np.arange(len(metadata.validation_losses))+1, metadata.validation_losses, label="Validation Loss")
    axes[0].set_ylabel(label)
    axes[0].legend()
    axes[0].set_yscale('log')

    axes[-1].plot(np.arange(len(metadata.mean_total_grad_norms))+1, metadata.mean_total_grad_norms)
    axes[-1].set_yscale('log')
    axes[-1].set_ylabel("Mean Gradient Norms")

    axes[-1].set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig('loss.pdf')

def main():
    train_dataset, val_dataset, test_dataset = load_and_split_dataset()
    print(f"Number of training samples: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=16)

    model = pileupNN(input_size=1024, hidden_size=256, output_size=1024)
    model.to(device)

    # model.load_state_dict(torch.load(config.DATA_NEURAL_NETWORK + "model_weights.pth", map_location="cpu"))

    # criterion = MSELoss()
    criterion = torch.nn.PoissonNLLLoss(log_input=False, full=True, reduction='sum')  # Exactly the same as Cash statistic?

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=16)


if __name__ == "__main__":
    main()