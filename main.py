import argparse
from data import load_mnist
from model import Net
import torch
import torch.optim as optim
import torch.nn.functional as F

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def evaluate(model, test_loader):
    test(model, test_loader)

def main():
    parser = argparse.ArgumentParser(description='MNIST PyTorch Example')
    parser.add_argument('--action', choices=['train', 'test', 'evaluate'], default='train', help='Choose action to perform (default: train)')
    args = parser.parse_args()

    train_loader, test_loader = load_mnist()
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    if args.action == 'train':
        for epoch in range(1, 6):
            train(model, train_loader, optimizer, epoch)
            test(model, test_loader)
    elif args.action == 'test':
        test(model, test_loader)
    elif args.action == 'evaluate':
        evaluate(model, test_loader)

if __name__ == '__main__':
    main()
