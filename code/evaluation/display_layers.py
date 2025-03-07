import torch
import matplotlib.pyplot as plt
import os

from data import load_and_split_dataset
from neuralnetwork import pileupNN
import config

plt.rcParams['text.usetex'] = True

activations = {}  # Dictionary to store activations

def get_activation_hook(layer_name):
    def hook_fn(module, input, output):
        activations[layer_name] = output.detach()  # Use detach() to avoid unnecessary computation graph retention
    return hook_fn

def plot_activations(model):
    train_dataset, val_dataset, test_dataset = load_and_split_dataset()
    index = 0
    input_data, target_data = test_dataset[index]
    input_fname, target_fname = test_dataset.get_filenames(index)

    hook_fc1 = model.fc1.register_forward_hook(get_activation_hook('fc1'))
    hook_fc2 = model.fc2.register_forward_hook(get_activation_hook('fc2'))
    hook_fc3 = model.fc3.register_forward_hook(get_activation_hook('penultimate_layer'))
    hook_fc4 = model.fc4.register_forward_hook(get_activation_hook('last_layer'))

    # Run the evaluation step (forward pass)
    model.eval()
    with torch.no_grad():
        predicted_output = model(input_data)

    # Extract the activations of the layers (after the forward pass)
    fc1_data = activations['fc1']
    fc2_data = activations['fc2']
    penultimate_data = activations['penultimate_layer']
    last_data = activations['last_layer']

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15/2.54, 20/2.54))
    fig.suptitle(os.path.basename(input_fname))

    axes[0].plot(input_data, color='black')
    axes[1].plot(fc1_data.cpu().numpy(), color='orange')
    axes[2].plot(fc2_data.cpu().numpy(), color='green')
    axes[3].plot(penultimate_data.cpu().numpy(), color='magenta')

    axes[-1].plot([0, len(last_data.cpu().numpy())], [0, 0], linestyle='--', color='gray')
    axes[-1].plot(last_data.cpu().numpy(), label='Activation before ReLU', color='blue', linewidth=2)
    axes[-1].plot(predicted_output, label='Predicted output (after ReLU)', color='red', linewidth=1)
    axes[-1].legend()

    axes[0].set_ylabel('Input data')
    axes[1].set_ylabel('First layer (fc1)')
    axes[2].set_ylabel('Second layer (fc2)')
    axes[3].set_ylabel('Penultimate layer (fc3)')
    axes[-1].set_ylabel('Last layer (fc4)')
    axes[-1].set_xlabel('Neuron Index')

    axes[0].set_xscale('log')
    axes[-1].set_xscale('log')

    outfile = "activations.pdf"
    plt.savefig(outfile)
    print(f"Wrote {outfile}")

    hook_fc1.remove()
    hook_fc2.remove()
    hook_fc3.remove()
    hook_fc4.remove()

def plot_weights_and_biases(model):
    weights_fc4 = model.fc4.weight.data.cpu().numpy()
    bias_fc4 = model.fc4.bias.data.cpu().numpy()

    nx = 4
    ny = int(len(weights_fc4)/4)
    fig, axes = plt.subplots(ncols=nx, nrows=ny+1, sharex=False, figsize=(nx*4, ny*2.2))
    fig.suptitle('Weights and biases of last layer (fc4)')
    fig.supxlabel('Neuron Index')
    fig.supylabel('Value')

    neuron_idx = 0
    for x in range(nx):
        for y in range(ny):
            axes[y, x].plot(weights_fc4[neuron_idx], linestyle='-', label=f'Weights (Neuron {neuron_idx})')
            axes[y, x].legend(loc='upper right')
            neuron_idx += 1

    axes[ny, 0].plot(bias_fc4, color="black", label="Bias")
    axes[ny, 0].legend(loc='upper right')
    [axes[ny, ii].set_axis_off() for ii in range(1, nx)]

    plt.tight_layout()
    plt.savefig("weights_and_biases.pdf")

def main():
    model = pileupNN()
    model.load_state_dict(torch.load(config.DATA_NEURAL_NETWORK + "model_weights.pth", map_location="cpu"))
    plot_activations(model)
    plot_weights_and_biases(model)


if __name__ == '__main__':
    main()