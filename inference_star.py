import argparse
import os

import joblib
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from models import *

def unnorm(img):
    return img * torch.Tensor((0.2023, 0.1994, 0.2010)).unsqueeze(1).unsqueeze(1) + torch.Tensor((0.4914, 0.4822, 0.4465)).unsqueeze(1).unsqueeze(1)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Caluculate Fisher')
parser.add_argument('--net', type=str, help='net path', required=True)
parser.add_argument('--fisher', type=str, help='fim path', required=True)
parser.add_argument('--cuda_device_number', type=int, default=0)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = torch.load(args.net)
fisher = Variable(torch.load(args.fisher).cuda(args.cuda_device_number))

if use_cuda:
    net.cuda(args.cuda_device_number)
    cudnn.benchmark = True

theta_star = nn.utils.parameters_to_vector([p.contiguous() for p in net.parameters()])
#dist_post = Normal(theta_star, 1 / (fisher + 1e-8))

net.eval()
saves = []
for batch_idx, (inputs, targets) in enumerate(testloader):
    if use_cuda:
        inputs, targets = inputs.cuda(args.cuda_device_number), targets.cuda(args.cuda_device_number)
    inputs, targets = Variable(inputs, volatile=True), Variable(targets)
    result = dict(input=inputs[0].data.cpu().numpy(), output=classes[targets[0].data.cpu().numpy()[0]])
    #inf_seq = []
    #for i in range(100):
    #    sampled_param = dist_post.sample()
    #    nn.utils.vector_to_parameters(sampled_param, nn.parameters())
    nn.utils.vector_to_parameters(theta_star, net.parameters())
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    result['inference'] = predicted[0]
    saves.append(result)
joblib.dump(saves, 'saves_star.pkl')

