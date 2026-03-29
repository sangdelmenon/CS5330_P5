# Sangeeth Deleep Menon
# CS5330 Project 5 - Extension: Gabor Filter Bank as Fixed First Layer
# Spring 2026
#
# Replaces the trained conv1 layer with a fixed bank of Gabor filters at
# different orientations. Only conv2 onward is trained. This tests whether
# hand-designed edge detectors can substitute for learned first-layer filters.

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from task1 import get_data, test_network, plot_accuracy


# creates n_filters Gabor kernels spanning orientations 0..pi
def make_gabor_bank(n_filters=10, kernel_size=5, sigma=1.8, lambd=4.5, gamma=0.5):
    orientations = np.linspace(0, np.pi, n_filters, endpoint=False)
    half = kernel_size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    filters = []
    for theta in orientations:
        xp =  x * np.cos(theta) + y * np.sin(theta)
        yp = -x * np.sin(theta) + y * np.cos(theta)
        gb = np.exp(-(xp ** 2 + gamma ** 2 * yp ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * xp / lambd)
        filters.append(gb)
    return np.array(filters, dtype=np.float32)


# CNN with conv1 replaced by a fixed Gabor filter bank
class GaborNetwork(nn.Module):

    def __init__(self, n_gabor=10):
        super(GaborNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, n_gabor, kernel_size=5)
        # initialise conv1 with Gabor weights and freeze them
        gabor = make_gabor_bank(n_gabor)
        self.conv1.weight = nn.Parameter(
            torch.from_numpy(gabor).unsqueeze(1), requires_grad=False)
        self.conv1.bias = nn.Parameter(
            torch.zeros(n_gabor), requires_grad=False)
        # trainable layers
        self.conv2 = nn.Conv2d(n_gabor, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# trains one epoch of the Gabor network, returns avg loss and accuracy
def train_epoch(model, loader, optimizer, epoch, device):
    model.train()
    total_loss, correct = 0.0, 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += output.argmax(1).eq(target).sum().item()
        if batch_idx % 200 == 0:
            print('  Epoch {} [{}/{}]: loss={:.4f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset), loss.item()))
    return total_loss / len(loader), 100. * correct / len(loader.dataset)


# visualises the 10 fixed Gabor filters as a 2x5 grid
def plot_gabor_filters(model):
    weights = model.conv1.weight.data[:, 0].cpu().numpy()
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(weights[i], cmap='gray', interpolation='none')
        ax.set_title('Gabor {}'.format(i), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('Fixed Gabor Filter Bank (conv1)')
    plt.tight_layout()
    plt.savefig('gabor_filters.png', dpi=150)
    plt.show()
    print('Saved gabor_filters.png')


# compares accuracy of the Gabor network against the standard trained network
def compare_with_standard(gabor_accs, standard_accs, n_epochs):
    epochs = list(range(1, n_epochs + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, gabor_accs,    'b-o', label='Gabor (fixed conv1)')
    plt.plot(epochs, standard_accs, 'r-o', label='Standard (learned conv1)')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Gabor Filter Bank vs. Learned Filters')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('gabor_vs_standard.png', dpi=150)
    plt.show()
    print('Saved gabor_vs_standard.png')


def main(argv):
    n_epochs = 5
    batch_size = 64
    lr = 0.01
    momentum = 0.5

    torch.manual_seed(42)

    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available()
                          else 'cpu')
    print('Using device:', device)

    train_loader, test_loader = get_data(batch_size)

    # build Gabor network and show its fixed filters
    model = GaborNetwork(n_gabor=10).to(device)
    print(model)
    plot_gabor_filters(model)

    # only train parameters that require gradients (conv2, fc1, fc2)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(trainable, lr=lr, momentum=momentum)

    gabor_test_accs = []
    for epoch in range(1, n_epochs + 1):
        print('Epoch {}:'.format(epoch))
        _, train_acc = train_epoch(model, train_loader, optimizer, epoch, device)
        _, test_acc = test_network(model, test_loader, device)
        gabor_test_accs.append(test_acc)
        print('  Train: {:.1f}%   Test: {:.1f}%'.format(train_acc, test_acc))

    torch.save(model.state_dict(), 'gabor_model.pth')
    print('Gabor model saved to gabor_model.pth')

    # load the standard trained model's per-epoch test accuracy for comparison
    # (run task1.py first; here we just retrain a fresh standard model for fairness)
    from task1 import MyNetwork, train_network
    std_model = MyNetwork().to(device)
    std_opt = optim.SGD(std_model.parameters(), lr=lr, momentum=momentum)
    std_test_accs = []
    for epoch in range(1, n_epochs + 1):
        train_network(std_model, train_loader, std_opt, epoch, device)
        _, acc = test_network(std_model, test_loader, device)
        std_test_accs.append(acc)

    compare_with_standard(gabor_test_accs, std_test_accs, n_epochs)

    print('\nFinal test accuracy:')
    print('  Gabor network:    {:.2f}%'.format(gabor_test_accs[-1]))
    print('  Standard network: {:.2f}%'.format(std_test_accs[-1]))

    return


if __name__ == '__main__':
    main(sys.argv)
