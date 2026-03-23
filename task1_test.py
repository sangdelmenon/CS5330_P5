# [Your Name]
# CS5330 Project 5 - Task 1 (E-F): Evaluate Trained CNN on Test Set and Handwritten Digits
# Spring 2026

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image

from task1 import MyNetwork


# loads the MNIST test set without shuffling
def get_test_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=10, shuffle=False)
    return test_loader


# loads the trained MNIST model from disk
def load_model(model_path='mnist_model.pth'):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


# runs the model on the first 10 test examples and prints results
def evaluate_first_ten(model, test_loader):
    data_iter = iter(test_loader)
    data, targets = next(data_iter)
    data, targets = data[:10], targets[:10]

    with torch.no_grad():
        output = model(data)

    print('\nFirst 10 test examples:')
    print('{:<4}  {:<60}  {:<9}  {}'.format('Idx', 'Network output (10 values)', 'Predicted', 'Correct'))
    print('-' * 90)
    for i in range(10):
        values = '  '.join(['{:.2f}'.format(v) for v in output[i]])
        pred = output[i].argmax().item()
        label = targets[i].item()
        print('{:<4}  {:<60}  {:<9}  {}'.format(i, values, pred, label))

    return data, targets, output


# plots the first 9 test examples in a 3x3 grid with predictions
def plot_predictions(data, targets, output):
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for i, ax in enumerate(axes.flat):
        pred = output[i].argmax().item()
        label = targets[i].item()
        ax.imshow(data[i][0], cmap='gray', interpolation='none')
        color = 'green' if pred == label else 'red'
        ax.set_title('Pred: {}  True: {}'.format(pred, label), color=color, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('First 9 MNIST Test Set Predictions')
    plt.tight_layout()
    plt.savefig('test_predictions.png', dpi=150)
    plt.show()
    print('Saved test_predictions.png')


# reads handwritten digit images from a directory and classifies them
# expects images named 0.png, 1.png, ..., 9.png (label is parsed from filename)
# MNIST digits are white on black - images are auto-inverted if they appear dark on light
def test_handwritten_digits(model, digits_dir='handwritten_digits'):
    if not os.path.exists(digits_dir):
        print('\nDirectory "{}" not found. Skipping handwritten digit test.'.format(digits_dir))
        print('Create the directory and add images named 0.png through 9.png.')
        return

    image_files = sorted([f for f in os.listdir(digits_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        print('No images found in "{}".'.format(digits_dir))
        return

    # transform matches MNIST preprocessing
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    tensors, raw_imgs, labels = [], [], []
    for fname in image_files:
        # extract digit label from filename (e.g. "3.png" or "digit_3.png")
        label = None
        for part in os.path.splitext(fname)[0].split('_'):
            if part.isdigit() and 0 <= int(part) <= 9:
                label = int(part)
                break

        img = Image.open(os.path.join(digits_dir, fname)).convert('L')
        img = img.resize((28, 28), Image.LANCZOS)
        arr = np.array(img)

        # invert if digit appears dark on light background (MNIST is white-on-black)
        if arr.mean() > 128:
            arr = 255 - arr
            img = Image.fromarray(arr)

        tensors.append(to_tensor(img))
        raw_imgs.append(arr)
        labels.append(label)

    batch = torch.stack(tensors)
    with torch.no_grad():
        output = model(batch)

    n = len(image_files)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2.5))
    axes_flat = np.array(axes).flat
    for i, ax in enumerate(axes_flat):
        if i < n:
            pred = output[i].argmax().item()
            title = 'Pred: {}'.format(pred)
            if labels[i] is not None:
                color = 'green' if pred == labels[i] else 'red'
                title += '\nTrue: {}'.format(labels[i])
            else:
                color = 'black'
            ax.imshow(raw_imgs[i], cmap='gray', interpolation='none')
            ax.set_title(title, color=color, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('Handwritten Digit Predictions')
    plt.tight_layout()
    plt.savefig('handwritten_results.png', dpi=150)
    plt.show()
    print('Saved handwritten_results.png')


# main function
def main(argv):
    model = load_model('mnist_model.pth')
    print('Model loaded.')
    print(model)

    test_loader = get_test_loader()

    # Task 1E: run on first 10 test examples
    data, targets, output = evaluate_first_ten(model, test_loader)
    plot_predictions(data, targets, output)

    # Task 1F: test on personal handwritten digits
    # Place images in a folder named 'handwritten_digits', named 0.png through 9.png
    test_handwritten_digits(model, digits_dir='handwritten_digits')

    return


if __name__ == "__main__":
    main(sys.argv)
