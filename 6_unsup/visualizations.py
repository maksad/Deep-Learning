import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
           'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def visualize_embeddings(encoder, dataloader, n_samples, device):
    n = 0
    codes, labels = [], []
    with torch.no_grad():
        for b_inputs, b_labels in dataloader:
            batch_size = b_inputs.size(0)
            b_codes = encoder(b_inputs.to(device))
            b_codes, b_labels = b_codes.cpu().data.numpy(), b_labels.cpu().data.numpy()
            if n + batch_size > n_samples:
                codes.append(b_codes[:n_samples-n])
                labels.append(b_labels[:n_samples-n])
                break
            else:
                codes.append(b_codes)
                labels.append(b_labels)
                n += batch_size
    codes = np.vstack(codes)
    if codes.shape[1] > 2:
        print('Use t-SNE')
        codes = TSNE().fit_transform(codes)
    labels = np.hstack(labels)

    colors = [
        'black', 'red', 'gold', 'palegreen', 'blue',
        'lightcoral', 'orange', 'mediumturquoise', 'dodgerblue', 'violet'
    ]
    fig, ax = plt.subplots(1)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height], which='both')
    for iclass in range(min(labels), max(labels)):
        ix = labels == iclass
        ax.plot(codes[ix, 0], codes[ix, 1], '.', color=colors[iclass])

    plt.legend(classes, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


def plot_images(images, n_rows=1):
    fig, axs = plt.subplots(n_rows, images.size(0) // n_rows)
    for ax, img in zip(axs.flat, images):
        ax.matshow(img[0].cpu().numpy(), cmap=plt.cm.Greys, vmin=0, vmax=1)
        ax.set_axis_off()
    plt.tight_layout(w_pad=0)


def visualize_reconstructions(encoder, decoder, dataloader, device):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    with torch.no_grad():
        encoder_outputs = encoder(images.to(device))
        decoder_outputs = decoder(encoder_outputs)
        reconstructions = decoder_outputs.cpu()
        images = images / 2 + 0.5  # inverse normalization
        reconstructions = reconstructions / 2 + 0.5  # inverse normalization
        plot_images(torch.cat([images, reconstructions]), n_rows=2)

