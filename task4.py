# [Your Name]
# CS5330 Project 5 - Task 4: MNIST Digit Recognition with a Transformer Network
# Spring 2026

import sys
import importlib.util
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from task1 import get_data


# loads NetTransformer and NetConfig from the template file
# (importlib is needed because the filename contains a hyphen)
def load_transformer_module():
    spec = importlib.util.spec_from_file_location(
        'net_transformer', 'NetTransformer-template.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# trains the transformer for one epoch, returns average loss and accuracy
def train_epoch(model, train_loader, optimizer, epoch, device):
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


# evaluates the transformer on the test set, returns average loss and accuracy
def test_epoch(model, test_loader, device):
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


# plots training and test accuracy curves and saves the figure
def plot_accuracy(train_accs, test_accs, n_epochs):
    epochs = list(range(1, n_epochs + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, 'b-o', label='Training Accuracy')
    plt.plot(epochs, test_accs, 'r-o', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Transformer: Training and Test Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('transformer_accuracy.png', dpi=150)
    plt.show()
    print('Saved transformer_accuracy.png')


# main function - trains and evaluates the transformer model on MNIST
def main(argv):
    net_module = load_transformer_module()
    NetConfig = net_module.NetConfig
    NetTransformer = net_module.NetTransformer

    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available()
                          else 'cpu')
    print('Using device:', device)

    config = NetConfig()
    torch.manual_seed(config.seed)

    model = NetTransformer(config).to(device)
    print('Transformer model:')
    print(model)
    print('\nConfig:')
    print(config.config_string)

    train_loader, test_loader = get_data(batch_size=config.batch_size)

    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)

    train_accs, test_accs = [], []
    for epoch in range(1, config.epochs + 1):
        print('Epoch {}:'.format(epoch))
        _, train_acc = train_epoch(model, train_loader, optimizer, epoch, device)
        _, test_acc = test_epoch(model, test_loader, device)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print('  Train: {:.1f}%   Test: {:.1f}%'.format(train_acc, test_acc))

    plot_accuracy(train_accs, test_accs, config.epochs)

    torch.save(model.state_dict(), 'transformer_model.pth')
    print('Transformer model saved to transformer_model.pth')

    return


if __name__ == "__main__":
    main(sys.argv)
