import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
# import matplotlib
# import matplotlib.pyplot as plt


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # convolutions:
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # here using two distinct layers to test
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
        # 6 input image channel, 16 output channels, 3x3 square convolution
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # Pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # activations
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        # flatten whole tensor
        self.flatten1 = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        # 1*28*28 -> 6*28*28
        x = self.conv1(x)
        # activation
        x = self.relu1(x)
        # 6*28*28 -> 6*14*14
        x = self.pool1(x)
        # 6*14*14 -> 16*12*12
        x = self.conv2(x)
        # activation
        x = self.relu2(x)
        # 16*12*12 -> 16*6*6
        x = self.pool2(x)
        # 16*6*6 -> 1*576
        x = self.flatten1(x)
        # 1*576 -> 1*120
        x = self.fc1(x)
        # activation
        x = self.relu3(x)
        # 1*120 -> 1*84
        x = self.fc2(x)
        # activation
        x = self.relu4(x)
        # 1*84 -> 10
        x = self.fc3(x)
        return x

    def getParameters(self):
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return (trainable_params, total_params)


def train(epoch):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Test Loss: {}'.format(loss.item()))


def test():
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = net(data)
            test_loss += F.nll_loss(output, target)
    test_loss = test_loss / len(test_loader.dataset)
    print('Test Loss: {}'.format(test_loss))


# All parameters:
seed = 42
n_epochs = 3
batch_size_train = 100
batch_size_test = 1000
learning_rate = 0.1
momentum = 0.5
log_interval = 10

torch.manual_seed(seed)
torch.backends.cudnn.enabled = False

net = Net()

# Data Loading:

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.1307,), (0.3081,))]
)

trainset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=batch_size_train,
    shuffle=True
)

testset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

test_loader = torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=batch_size_test,
    shuffle=False
)

# show example

# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# print(example_data.shape)
# print(example_targets.shape)

# show single image
# image = example_data[0]
# plt.imshow(torchvision.transforms.ToPILImage()(image), cmap='gray')
# plt.show()
# print(example_targets[0])

(trainParams, totalParams) = net.getParameters()
print('parameters: {}/{} are trainable'.format(trainParams, totalParams))

optimizer = optim.SGD(params=net.parameters(), lr=learning_rate, momentum=momentum)

train(1)

# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# fig

# target = torch.randn(10)  # a dummy target, for example
# target = torch.zeros([1,10],dtype=torch.float32)
# target[0,0] = 1
# target = target.view(1, -1)  # make it the same shape as output


# for epoch in range(0,1):
#   optimizer.zero_grad()
#   output=net(input)
#   target = target.view(1, -1)  # make it the same shape as output
#   loss=criterion(output,target)
#   print(loss.grad_fn)
#   print(loss)
#   loss.backward()
#   optimizer.step()

# print(net(input))