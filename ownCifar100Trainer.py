import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from collections import namedtuple

Statistics = namedtuple('Statistics', ['loss', 'accuracy'])


class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
        # convolutions:
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)
        # fully connections
        self.fc1 = nn.Linear(16 * 6 * 6, 250)
        self.fc2 = nn.Linear(250, 150)
        self.fc3 = nn.Linear(150, 100)
        self.flatten1 = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.flatten1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def getParameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (trainable_params, total_params)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx % args.log_interval == 0) and (not args.no_logging):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, stats):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    correct_percentage = 100. * correct / len(test_loader.dataset)
    if (not args.no_logging):
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            correct_percentage))
    stats.append(Statistics(test_loss, correct_percentage))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--no-logging', action='store_true', default=False,
                        help='No logging during training and testing')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    print('use Cuda? {}'.format(use_cuda))

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=False,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=False,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        **kwargs
    )

    model = Net1().to(device)
    (trainParams, totalParams) = model.getParameters()
    if (not args.no_logging):
        print('Parameters: {}/{} are trainable'.format(trainParams, totalParams))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    stats = []

    test(args, model, device, test_loader, stats)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(1, 61):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, stats)

    optimizer = optim.SGD(model.parameters(), lr=0.02)
    for epoch in range(1, 61):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, stats)

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    for epoch in range(1, 41):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, stats)
    optimizer = optim.SGD(model.parameters(), lr=0.0008)

    for epoch in range(1, 41):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, stats)
    # for epoch in range(1, args.epochs + 1):
    #     train(args, model, device, train_loader, optimizer, epoch)
    #     test(args, model, device, test_loader, stats)
    #     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if (args.save_model):
        torch.save(model.state_dict(), "ownNet.pt")

    f1 = open('stats1_own.txt', 'a+')
    f2 = open('stats2_own.txt', 'a+')
    print('Statistics:\nlr: {} momentum: {}'.format(args.lr, args.momentum))
    f1.write('lr: {} momentum: {}\n'.format(args.lr, args.momentum))
    f2.write('{},{},{},{}\n'.format(args.lr, args.momentum, stats[-1].loss, stats[-1].accuracy))
    for s in stats:
        print('loss = {}, acc = {}%'.format(s.loss, s.accuracy))
        f1.write('{:.4f}, {:.4f}\n'.format(s.loss, s.accuracy))
    f1.close()
    f2.close()


if __name__ == '__main__':
    main()
