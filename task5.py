# [Your Name]
# CS5330 Project 5 - Task 5: Architecture Experiment
# Spring 2026
#
# Evaluates 64 CNN configurations by varying three architectural dimensions:
#   Dim 1 - conv1 filter count:  [5, 10, 20, 40]
#   Dim 2 - fc1 hidden nodes:    [25, 50, 100, 200]
#   Dim 3 - dropout rate:        [0.1, 0.3, 0.5, 0.7]
#
# Hypotheses (fill in before running):
#   Dim 1: More conv1 filters should capture richer edge patterns -> higher accuracy.
#   Dim 2: Larger fc1 should improve accuracy up to a point before overfitting.
#   Dim 3: Very low dropout may overfit; very high dropout may underfit. 0.3-0.5 optimal.

import sys
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from task1 import get_data


# configurable CNN - same architecture as MyNetwork but with variable dimensions
class ConfigurableNetwork(nn.Module):

    def __init__(self, conv1_filters=10, fc1_nodes=50, dropout_rate=0.5):
        super(ConfigurableNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_filters, kernel_size=5)
        self.conv2 = nn.Conv2d(conv1_filters, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout(dropout_rate)
        # conv2 output is always 20 channels, 4x4 spatial = 320 features
        self.fc1 = nn.Linear(320, fc1_nodes)
        self.fc2 = nn.Linear(fc1_nodes, 10)

    # computes a forward pass through the configurable network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# trains a model for n_epochs and returns the final test accuracy
def run_experiment(conv1_filters, fc1_nodes, dropout_rate,
                   train_loader, test_loader, device,
                   n_epochs=3, lr=0.01, momentum=0.5):
    model = ConfigurableNetwork(
        conv1_filters=conv1_filters,
        fc1_nodes=fc1_nodes,
        dropout_rate=dropout_rate
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, n_epochs + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    # evaluate on test set
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


# runs the full grid search and saves results to a CSV file
def run_grid_search(train_loader, test_loader, device):
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
                acc = run_experiment(conv1_f, fc1_n, drop, train_loader, test_loader, device)
                print('accuracy={:.2f}%'.format(acc))
                results.append({
                    'conv1_filters': conv1_f,
                    'fc1_nodes': fc1_n,
                    'dropout': drop,
                    'test_accuracy': acc
                })

    # save results to CSV
    csv_path = 'experiment_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['conv1_filters', 'fc1_nodes', 'dropout', 'test_accuracy'])
        writer.writeheader()
        writer.writerows(results)
    print('\nResults saved to', csv_path)

    return results


# plots the effect of each dimension on test accuracy (marginal averages)
def plot_results(results):
    import numpy as np

    conv1_vals   = sorted(set(r['conv1_filters'] for r in results))
    fc1_vals     = sorted(set(r['fc1_nodes']     for r in results))
    dropout_vals = sorted(set(r['dropout']        for r in results))

    def mean_acc(key, val):
        subset = [r['test_accuracy'] for r in results if r[key] == val]
        return np.mean(subset)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(conv1_vals, [mean_acc('conv1_filters', v) for v in conv1_vals], 'b-o')
    axes[0].set_xlabel('Conv1 Filters')
    axes[0].set_ylabel('Mean Test Accuracy (%)')
    axes[0].set_title('Dim 1: Conv1 Filter Count')
    axes[0].grid(True)

    axes[1].plot(fc1_vals, [mean_acc('fc1_nodes', v) for v in fc1_vals], 'g-o')
    axes[1].set_xlabel('FC1 Hidden Nodes')
    axes[1].set_title('Dim 2: FC1 Hidden Nodes')
    axes[1].grid(True)

    axes[2].plot(dropout_vals, [mean_acc('dropout', v) for v in dropout_vals], 'r-o')
    axes[2].set_xlabel('Dropout Rate')
    axes[2].set_title('Dim 3: Dropout Rate')
    axes[2].grid(True)

    plt.suptitle('Effect of Architecture Dimensions on Test Accuracy (averaged over other dims)')
    plt.tight_layout()
    plt.savefig('experiment_plots.png', dpi=150)
    plt.show()
    print('Saved experiment_plots.png')


# prints the top 10 and bottom 5 configurations by test accuracy
def print_summary(results):
    sorted_results = sorted(results, key=lambda r: r['test_accuracy'], reverse=True)
    print('\n--- Top 10 configurations ---')
    for r in sorted_results[:10]:
        print('  conv1={:<3}  fc1={:<4}  dropout={:.1f}  acc={:.2f}%'.format(
            r['conv1_filters'], r['fc1_nodes'], r['dropout'], r['test_accuracy']))
    print('\n--- Bottom 5 configurations ---')
    for r in sorted_results[-5:]:
        print('  conv1={:<3}  fc1={:<4}  dropout={:.1f}  acc={:.2f}%'.format(
            r['conv1_filters'], r['fc1_nodes'], r['dropout'], r['test_accuracy']))


# main function
def main(argv):
    torch.manual_seed(42)

    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available()
                          else 'cpu')
    print('Using device:', device)
    print('Running 64 configurations (4 conv1 x 4 fc1 x 4 dropout), 3 epochs each.\n')

    train_loader, test_loader = get_data(batch_size=64)

    results = run_grid_search(train_loader, test_loader, device)
    print_summary(results)
    plot_results(results)

    return


if __name__ == "__main__":
    main(sys.argv)
