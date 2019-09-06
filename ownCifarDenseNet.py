import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets, models
from collections import namedtuple

Statistics = namedtuple('Statistics', ['loss', 'accuracy'])


class Net(models.resnet.ResNet):

    def __init__(self):
        super(Net, self).__init__(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=100)

    def forward(self, x):
        return F.softmax(
            super(Net, self).forward(x),
            dim=-1
        )


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
            # data.unsqueeze(0)
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
    parser = argparse.ArgumentParser(description='PyTorch CIFAR 100 Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                        help='input batch size for testing (default: 50)')
    parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
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

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    trainset = datasets.CIFAR100(
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
    testset = datasets.CIFAR100(
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

    model = torch.hub.load('pytorch/vision', 'densenet121', pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    if (not args.no_logging):
        print('Parameters: {}'.format(total_params))

    stats = []

    optimizer = optim.Adadelta(model.parameters(), lr=0.1)
    test(args, model, device, test_loader, stats)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, stats)

    if (args.save_model):
        torch.save(model.state_dict(), "myDenseNet.pt")
    f1 = open('stats1_dense.txt', 'a+')
    f2 = open('stats2_dense.txt', 'a+')
    print('Statistics:\n')
    f2.write('{},{}\n'.format(stats[-1].loss, stats[-1].accuracy))
    for s in stats:
        print('loss = {}, acc = {}%'.format(s.loss, s.accuracy))
        f1.write('{:.4f}, {:.4f}\n'.format(s.loss, s.accuracy))
    f1.close()
    f2.close()


if __name__ == '__main__':
    main()
