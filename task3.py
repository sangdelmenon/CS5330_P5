# [Your Name]
# CS5330 Project 5 - Task 3: Transfer Learning for Greek Letter Recognition
# Spring 2026
#
# Before running: unzip greek_train.zip to create the greek_train/ directory
# Place personal Greek letter images in personal_greek/alpha/, /beta/, /gamma/

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from task1 import MyNetwork


# transforms Greek letter images (133x133 RGB) to match MNIST format (28x28 grayscale, white-on-black)
class GreekTransform:

    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# loads the Greek letter dataset from the given directory using ImageFolder
def get_greek_data(training_set_path, batch_size=5):
    greek_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            training_set_path,
            transform=transforms.Compose([
                transforms.ToTensor(),
                GreekTransform(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=batch_size,
        shuffle=True)
    return greek_loader


# loads the pretrained MNIST model and modifies it for 3-class Greek letter recognition
def build_greek_model(model_path='mnist_model.pth'):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # freeze all parameters so only the new last layer trains
    for param in model.parameters():
        param.requires_grad = False

    # replace fc2 (50 -> 10) with a 3-class output layer
    model.fc2 = nn.Linear(50, 3)

    return model


# trains the modified model on Greek letter data, returns per-epoch losses
def train_greek(model, greek_loader, n_epochs=30, lr=0.01):
    # only fc2 has requires_grad=True, so only it gets updated
    optimizer = optim.SGD(model.fc2.parameters(), lr=lr, momentum=0.5)
    losses = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for data, target in greek_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
        avg_loss = epoch_loss / len(greek_loader)
        accuracy = 100. * correct / total
        losses.append(avg_loss)
        print('Epoch {:>3}/{}: loss={:.4f}  accuracy={}/{} ({:.1f}%)'.format(
            epoch, n_epochs, avg_loss, correct, total, accuracy))

    return losses


# plots the training loss curve and saves the figure
def plot_training_loss(losses):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses) + 1), losses, 'b-o')
    plt.xlabel('Epoch')
    plt.ylabel('NLL Loss')
    plt.title('Greek Letter Transfer Learning - Training Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('greek_training_loss.png', dpi=150)
    plt.show()
    print('Saved greek_training_loss.png')


# evaluates the trained Greek model on all provided examples
def evaluate_greek(model, greek_loader, class_names):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in greek_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
    print('\nFinal accuracy on Greek training set: {}/{} ({:.1f}%)'.format(
        correct, total, 100. * correct / total))


# tests the model on personal handwritten Greek letter images
def test_personal_greek(model, class_names, images_dir='personal_greek'):
    if not os.path.exists(images_dir):
        print('\nDirectory "{}" not found. Skipping personal Greek letter test.'.format(images_dir))
        print('Create it with subfolders alpha/, beta/, gamma/ containing your images (~128x128).')
        return

    loader = get_greek_data(images_dir, batch_size=50)
    model.eval()

    imgs_list, preds_list, labels_list = [], [], []
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            preds = output.argmax(dim=1)
            for i in range(len(data)):
                imgs_list.append(data[i, 0].numpy())
                preds_list.append(preds[i].item())
                labels_list.append(target[i].item())

    n = len(imgs_list)
    ncols = min(6, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2.5))
    axes_flat = np.array(axes).flat
    for i, ax in enumerate(axes_flat):
        if i < n:
            pred_name = class_names[preds_list[i]]
            true_name = class_names[labels_list[i]]
            color = 'green' if preds_list[i] == labels_list[i] else 'red'
            ax.imshow(imgs_list[i], cmap='gray')
            ax.set_title('P: {}\nT: {}'.format(pred_name, true_name), color=color, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('Personal Greek Letter Predictions')
    plt.tight_layout()
    plt.savefig('personal_greek_results.png', dpi=150)
    plt.show()
    print('Saved personal_greek_results.png')


# main function
def main(argv):
    training_set_path = 'greek_train'

    if not os.path.exists(training_set_path):
        print('Error: "{}" not found.'.format(training_set_path))
        print('Run:  unzip greek_train.zip')
        return

    # build the modified model
    model = build_greek_model('mnist_model.pth')
    print('Modified network (last layer replaced for 3-class output):')
    print(model)

    greek_loader = get_greek_data(training_set_path)

    # class names are determined by ImageFolder's folder sort order
    class_names = sorted(os.listdir(training_set_path))
    class_names = [c for c in class_names if not c.startswith('.')]
    print('Classes:', class_names)

    # train
    losses = train_greek(model, greek_loader, n_epochs=30)
    plot_training_loss(losses)
    evaluate_greek(model, greek_loader, class_names)

    # save the Greek model
    torch.save(model.state_dict(), 'greek_model.pth')
    print('Greek model saved to greek_model.pth')

    # test on personal Greek letter images
    test_personal_greek(model, class_names, images_dir='personal_greek')

    return


if __name__ == "__main__":
    main(sys.argv)
