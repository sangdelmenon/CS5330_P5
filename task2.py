# [Your Name]
# CS5330 Project 5 - Task 2: Examine the Trained Network
# Spring 2026

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from task1 import MyNetwork, get_data


# loads the trained MNIST model from disk
def load_model(model_path='mnist_model.pth'):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


# prints and visualizes the 10 conv1 filters in a 3x4 grid
def analyze_first_layer(model):
    with torch.no_grad():
        weights = model.conv1.weight  # shape: [10, 1, 5, 5]

    print('conv1 weight shape:', weights.shape)
    print('conv1 filter values:')
    for i in range(10):
        print('  Filter {}: {}'.format(i, weights[i, 0].numpy()))

    fig, axes = plt.subplots(3, 4, figsize=(8, 7))
    for i in range(10):
        ax = axes[i // 4][i % 4]
        filter_img = weights[i, 0].numpy()
        ax.imshow(filter_img, cmap='viridis', interpolation='none')
        ax.set_title('Filter {}'.format(i), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    # hide the two unused subplots
    axes[2][2].set_visible(False)
    axes[2][3].set_visible(False)
    plt.suptitle('conv1 Filters (10 filters, 5x5)')
    plt.tight_layout()
    plt.savefig('conv1_filters.png', dpi=150)
    plt.show()
    print('Saved conv1_filters.png')

    return weights


# applies the 10 conv1 filters to the first training image using cv2.filter2D
def show_filter_effects(model, train_loader):
    data_iter = iter(train_loader)
    data, targets = next(data_iter)
    # use the first image in the batch (shape: 28x28)
    first_image = data[0, 0].numpy()
    print('\nApplying filters to training example with label:', targets[0].item())

    with torch.no_grad():
        weights = model.conv1.weight.numpy()  # shape: [10, 1, 5, 5]

    # 3x4 grid: original image + 10 filtered images + 1 empty slot
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))

    axes[0][0].imshow(first_image, cmap='gray', interpolation='none')
    axes[0][0].set_title('Original\n(label={})'.format(targets[0].item()), fontsize=9)
    axes[0][0].set_xticks([])
    axes[0][0].set_yticks([])

    for i in range(10):
        row = (i + 1) // 4
        col = (i + 1) % 4
        kernel = weights[i, 0]  # 5x5 filter
        filtered = cv2.filter2D(first_image, ddepth=-1, kernel=kernel)
        axes[row][col].imshow(filtered, cmap='gray', interpolation='none')
        axes[row][col].set_title('Filter {}'.format(i), fontsize=9)
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])

    # hide last empty subplot
    axes[2][3].set_visible(False)
    plt.suptitle('Effect of conv1 Filters on First Training Image')
    plt.tight_layout()
    plt.savefig('conv1_filter_effects.png', dpi=150)
    plt.show()
    print('Saved conv1_filter_effects.png')


# main function
def main(argv):
    model = load_model('mnist_model.pth')
    print('Network structure:')
    print(model)

    train_loader, _ = get_data(batch_size=64)

    # Task 2A: analyze and visualize conv1 filters
    analyze_first_layer(model)

    # Task 2B: apply filters to first training image
    show_filter_effects(model, train_loader)

    return


if __name__ == "__main__":
    main(sys.argv)
