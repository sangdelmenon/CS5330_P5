# [Your Name]
# CS5330 Project 5 - Task 1: Build and Train a CNN for MNIST Digit Recognition
# Spring 2026

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# CNN network for MNIST digit recognition
class MyNetwork(nn.Module):

    def __init__(self):
        super(MyNetwork, self).__init__()
        # conv layer: 10 5x5 filters, input 1 channel
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # conv layer: 20 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # dropout layer with 50% rate
        self.conv2_drop = nn.Dropout(0.5)
        # fully connected: 320 -> 50
        self.fc1 = nn.Linear(320, 50)
        # output layer: 50 -> 10
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass through the network
    def forward(self, x):
        # conv1 -> max pool 2x2 -> ReLU
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # conv2 -> dropout -> max pool 2x2 -> ReLU
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # flatten: 20 * 4 * 4 = 320
        x = x.view(-1, 320)
        # fc1 -> ReLU
        x = F.relu(self.fc1(x))
        # fc2 -> log_softmax
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# loads MNIST training and test data loaders
def get_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    # test set is never shuffled so examples are always in the same order
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# plots the first 6 examples from the test set and saves the figure
def visualize_test_examples(test_loader):
    data_iter = iter(test_loader)
    example_data, example_targets = next(data_iter)
    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(example_data[i][0], cmap='gray', interpolation='none')
        ax.set_title('Label: {}'.format(example_targets[i].item()))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('First 6 MNIST Test Examples')
    plt.tight_layout()
    plt.savefig('test_examples.png', dpi=150)
    plt.show()
    print('Saved test_examples.png')


# trains the network for one epoch, returns average loss and accuracy
def train_network(model, train_loader, optimizer, epoch, device):
    model.train()
    total_loss = 0.0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        if batch_idx % 200 == 0:
            print('  Epoch {} [{}/{}]: loss={:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    return avg_loss, accuracy


# evaluates the network on the test set, returns average loss and accuracy
def test_network(model, test_loader, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('  Test:  loss={:.4f}, accuracy={}/{} ({:.1f}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy


# plots training and test accuracy per epoch and saves the figure
def plot_accuracy(train_accuracies, test_accuracies, n_epochs):
    epochs = list(range(1, n_epochs + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracies, 'b-o', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'r-o', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('CNN Training and Test Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    print('Saved training_curves.png')


# main function - builds, trains, and saves the MNIST CNN model
def main(argv):
    n_epochs = 5
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.5

    torch.manual_seed(42)

    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available()
                          else 'cpu')
    print('Using device:', device)

    train_loader, test_loader = get_data(batch_size)

    # Task 1A: visualize first 6 test examples
    visualize_test_examples(test_loader)

    # Task 1B: build the CNN model
    model = MyNetwork().to(device)
    print(model)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_accuracies = []
    test_accuracies = []

    # Task 1C: train for n_epochs and record accuracy after each epoch
    for epoch in range(1, n_epochs + 1):
        print('Epoch {}:'.format(epoch))
        _, train_acc = train_network(model, train_loader, optimizer, epoch, device)
        _, test_acc = test_network(model, test_loader, device)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        print('  Train: {:.1f}%   Test: {:.1f}%'.format(train_acc, test_acc))

    plot_accuracy(train_accuracies, test_accuracies, n_epochs)

    # Task 1D: save the trained model
    torch.save(model.state_dict(), 'mnist_model.pth')
    print('Model saved to mnist_model.pth')

    return


if __name__ == "__main__":
    main(sys.argv)
