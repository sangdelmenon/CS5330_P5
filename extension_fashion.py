# Sangeeth Deleep Menon
# CS5330 Project 5 - Extension: Architecture Experiment on Fashion-MNIST
# Spring 2026
#
# Repeats the Task 5 grid search on the Fashion-MNIST dataset to see whether
# the same architectural trends (more filters, wider fc1, moderate dropout)
# hold for a harder 10-class classification problem.
#
# Fashion-MNIST classes:
#   0=T-shirt/top  1=Trouser  2=Pullover  3=Dress    4=Coat
#   5=Sandal       6=Shirt    7=Sneaker   8=Bag      9=Ankle boot

import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from task5 import ConfigurableNetwork, run_experiment, plot_results, print_summary


# returns Fashion-MNIST train and test loaders
def get_fashion_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST mean/std
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, transform=transform),
        batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# runs the full 4x4x4 grid search on Fashion-MNIST
def run_fashion_grid_search(train_loader, test_loader, device):
    conv1_options  = [5, 10, 20, 40]
    fc1_options    = [25, 50, 100, 200]
    dropout_options = [0.1, 0.3, 0.5, 0.7]

    results = []
    total = len(conv1_options) * len(fc1_options) * len(dropout_options)
    run = 0

    for conv1_f in conv1_options:
        for fc1_n in fc1_options:
            for drop in dropout_options:
                run += 1
                print('Run {}/{}: conv1_filters={}, fc1_nodes={}, dropout={}'.format(
                    run, total, conv1_f, fc1_n, drop), end='  ', flush=True)
                acc = run_experiment(conv1_f, fc1_n, drop,
                                     train_loader, test_loader, device, n_epochs=3)
                print('accuracy={:.2f}%'.format(acc))
                results.append({
                    'conv1_filters': conv1_f,
                    'fc1_nodes':     fc1_n,
                    'dropout':       drop,
                    'test_accuracy': acc
                })

    csv_path = 'fashion_experiment_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['conv1_filters', 'fc1_nodes', 'dropout', 'test_accuracy'])
        writer.writeheader()
        writer.writerows(results)
    print('\nResults saved to', csv_path)

    return results


# plots side-by-side marginal accuracy curves for MNIST and Fashion-MNIST
def compare_datasets(mnist_results, fashion_results):
    dims = [
        ('conv1_filters', 'Conv1 Filters',   [5, 10, 20, 40]),
        ('fc1_nodes',     'FC1 Hidden Nodes', [25, 50, 100, 200]),
        ('dropout',       'Dropout Rate',     [0.1, 0.3, 0.5, 0.7]),
    ]

    def mean_acc(results, key, val):
        return np.mean([r['test_accuracy'] for r in results if r[key] == val])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = [('b', 'c'), ('g', 'lime'), ('r', 'salmon')]

    for ax, (key, label, vals), (c_mnist, c_fashion) in zip(axes, dims, colors):
        ax.plot(vals, [mean_acc(mnist_results,   key, v) for v in vals],
                color=c_mnist,   marker='o', label='MNIST')
        ax.plot(vals, [mean_acc(fashion_results, key, v) for v in vals],
                color=c_fashion, marker='s', linestyle='--', label='Fashion-MNIST')
        ax.set_xlabel(label)
        ax.set_ylabel('Mean Test Accuracy (%)')
        ax.set_title(label)
        ax.legend()
        ax.grid(True)

    plt.suptitle('MNIST vs Fashion-MNIST: Effect of Architecture on Test Accuracy')
    plt.tight_layout()
    plt.savefig('fashion_vs_mnist.png', dpi=150)
    plt.show()
    print('Saved fashion_vs_mnist.png')


def main(argv):
    torch.manual_seed(42)

    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available()
                          else 'cpu')
    print('Using device:', device)
    print('Running 64 configurations on Fashion-MNIST (4 conv1 x 4 fc1 x 4 dropout), 3 epochs each.\n')

    train_loader, test_loader = get_fashion_data(batch_size=64)

    fashion_results = run_fashion_grid_search(train_loader, test_loader, device)
    print_summary(fashion_results)
    plot_results(fashion_results)

    # load MNIST results from task5 CSV for comparison (if it exists)
    try:
        mnist_results = []
        with open('experiment_results.csv', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mnist_results.append({
                    'conv1_filters': int(row['conv1_filters']),
                    'fc1_nodes':     int(row['fc1_nodes']),
                    'dropout':       float(row['dropout']),
                    'test_accuracy': float(row['test_accuracy'])
                })
        import csv as _csv
        compare_datasets(mnist_results, fashion_results)
    except FileNotFoundError:
        print('\nNo experiment_results.csv found — run task5.py first for the MNIST comparison plot.')

    return


if __name__ == '__main__':
    import csv
    main(sys.argv)
