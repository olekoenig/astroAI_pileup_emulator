import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def plot_loss(metadata, label = "Loss"):
    fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(12/2.54, 12/2.54))
    # Note that running training loss (not plotted here) should start half an epoch
    # earlier (at 0.5) because it is calculated during training while the gradients
    # are still updated.
    #axes[0].plot(np.arange(len(metadata.training_losses))+1, metadata.training_losses, label="Training Loss")
    axes[0].plot(np.arange(len(metadata.validation_losses))+1, metadata.validation_losses, label="Validation Loss", linewidth=3)

    axes[0].plot(np.arange(len(metadata.term1))+1, metadata.term1+metadata.term2, label=r"Manual total loss")
    axes[0].plot(np.arange(len(metadata.term1))+1, metadata.term1, label=r"Term1: $\frac{1}{2}\log(\sigma^2)$")
    axes[0].plot(np.arange(len(metadata.term2)) + 1, metadata.term2, label=r"Term2: $\frac{1}{2}(\mu-\hat{\mu})^2/\sigma^2$")
    axes[0].set_ylabel(label)
    axes[0].legend()
    # axes[0].set_yscale('log')

    axes[-1].plot(np.arange(len(metadata.mean_total_grad_norms))+1, metadata.mean_total_grad_norms)
    axes[-1].set_yscale('log')
    axes[-1].set_ylabel("Mean Gradient Norms")

    axes[-1].set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig('loss.pdf')
