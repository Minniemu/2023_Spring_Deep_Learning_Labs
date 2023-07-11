from dataloader import RetinopathyLoader
from plot import plt_confusion_matrix
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
import os
from tqdm import tqdm
from torch import cuda


def downsample(in_ch, out_ch, stride):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=stride, bias=False),
        nn.BatchNorm2d(out_ch))

# BasicBlock
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_stride):
        super(BasicBlock, self).__init__()
        if downsample_stride is None:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.downsample = None
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.downsample = downsample(in_channels, out_channels, downsample_stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        ori = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            ori = self.downsample(ori)
        out = self.relu(out+ori)
        return out

# Bottleneck
class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, downsample_stride):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        if downsample_stride is None:
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.downsample = None
        else:
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 3), stride=downsample_stride, padding=(1, 1), bias=False)
            self.downsample = downsample(in_channels, out_channels, downsample_stride)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        ori = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            ori = self.downsample(ori)
        out = self.relu(out+ori)
        return out
# Resnet18
class ResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2,2), padding=(3,3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, None),
            BasicBlock(64, 64, None))
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, (2,2)),
            BasicBlock(128, 128, None))
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, (2, 2)),
            BasicBlock(256, 256, None))
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, (2, 2)),
            BasicBlock(512, 512, None))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 5)
    
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.maxpool(outputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.avgpool(outputs)
        outputs = self.fc(outputs.reshape(outputs.shape[0], -1))
        return outputs
# Resnet50
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, 256, (1, 1)),
            Bottleneck(256, 64, 256, None),
            Bottleneck(256, 64, 256, None))
        self.layer2 = nn.Sequential(
            Bottleneck(256, 128, 512, (2, 2)),
            Bottleneck(512, 128, 512, None),
            Bottleneck(512, 128, 512, None),
            Bottleneck(512, 128, 512, None))
        self.layer3 = nn.Sequential(
            Bottleneck(512, 256, 1024, (2, 2)),
            Bottleneck(1024, 256, 1024, None),
            Bottleneck(1024, 256, 1024, None),
            Bottleneck(1024, 256, 1024, None),
            Bottleneck(1024, 256, 1024, None),
            Bottleneck(1024, 256, 1024, None))
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512, 2048, (2, 2)),
            Bottleneck(2048, 512, 2048, None),
            Bottleneck(2048, 512, 2048, None))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, 5)
            
    def forward(self, x):
        outputs = self.relu(self.bn1(self.conv1(x)))
        outputs = self.maxpool(outputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.avgpool(outputs)
        outputs = self.fc(outputs.reshape(outputs.shape[0], -1))
        return outputs



def train(model_type, device, model, train_loader, test_loader, optimizer, criterion, epoch_num, ckpt_path):
    os.makedirs(ckpt_path, exist_ok=True)
    scheduler = StepLR(optimizer, step_size= 50, gamma=0.95)
    batch_count = 0
    epoch_pbar = tqdm(range(1, epoch_num+1))
    for epoch in epoch_pbar:
        model.to(device)
        model.train()
        epoch_loss = 0
        correct = 0.0
        total = 0.0
        avg_loss = 0.0
        batch_pbar = tqdm(train_loader)
        for i, (images, labels) in enumerate(batch_pbar):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_count += 1
            epoch_loss += loss.item()
            batch_pbar.set_description(f'[train] [epoch:{epoch:>4}/{epoch_num}] [batch: {i+1:>5}/{len(train_loader)}] loss: {loss.item():.4f}')
        scheduler.step()
        epoch_pbar.set_description(f'[train] [epoch:{epoch:>4}/{epoch_num}] loss: {epoch_loss/len(train_loader):.4f}')
        acc = 100 * correct / total
        avg_loss /= len(train_loader)
        print('Train accuracy : {:.2f} %, Train loss : {:.4f}'.format(acc, avg_loss))
        torch.save(model.state_dict(), f'{ckpt_path}{model_type}_epoch{epoch}.ckpt')
        evaluate(model_type, model, device, test_loader, criterion, epoch)

def evaluate(model_type, model, device, test_loader, criterion, epoch):
    predict = []
    ground_truth = []
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        avg_loss = 0
        batch_pbar = tqdm(test_loader)
        for i, (images, labels) in enumerate(batch_pbar):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            avg_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            predict += predicted.cpu().numpy().tolist()
            ground_truth += labels.cpu().numpy().tolist()
            correct += (predicted == labels).sum().item()
            batch_pbar.set_description(f'[test] [batch: {i+1:>5}/{len(test_loader)}] loss: {loss.item():.4f}')
        avg_loss /= len(test_loader)
        acc = 100 * correct / total
        print('Test accuracy : {:.2f} %, Test loss : {:.4f}'.format(acc, avg_loss))
        plt_confusion_matrix(ground_truth, predict, title=model_type+'_'+str(epoch), normalize='true')


def main():
    # Define hyper-parameters
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--ckpt_path', type=str, default='./model/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    
    print("output path = ", args.ckpt_path)
    # Assign device
    device = args.device if torch.cuda.is_available() else 'cpu'
    model_type = args.model
    # Load data
    train_data = RetinopathyLoader(root='./data/new_train', mode='train')
    test_data = RetinopathyLoader (root='./data/new_test', mode='test')
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, pin_memory_device=device)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True, pin_memory_device=device)

    # Define model
    if args.model == 'resnet18':
        model = models.resnet18(pretrained=args.pretrained)
        filters = model.fc.in_features
        model.fc = nn.Linear(filters, 5)
    elif args.model == 'resnet18_wo_pretrained':
        model = ResNet18()
    elif args.model == 'resnet50':
        model = models.resnet50(pretrained=args.pretrained)
        filters = model.fc.in_features
        model.fc = nn.Linear(filters, 5)
    elif args.model == 'resnet50_wo_pretrained':
        model = ResNet50()
    
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam': 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 
    elif args.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    model.to(device)
    print(model)
    if args.mode =='train':
        # Train model
        train(args.model, device, model, train_loader, test_loader, optimizer, criterion, epoch_num=args.epoch_num, ckpt_path=args.ckpt_path)
    elif args.mode == 'test':
        if args.model == 'resnet18':
            model.load_state_dict(torch.load('./model/resnet18/resnet18_pretrained_epoch9.ckpt'))
        elif args.model =='resnet18_wo_pretrained':
            model.load_state_dict(torch.load('./model/resnet18wo/resnet18_wo_pretrained_epoch9.ckpt'))
        elif args.model == 'resnet50':
            model.load_state_dict(torch.load('./model/resnet50/resnet50_pretrained_epoch9.ckpt'))
        elif args.model == 'resnet50_wo_pretrained':
            model.load_state_dict(torch.load('./model/resnet50wo/resnet50_wo_pretrained_epoch8.ckpt'))
        evaluate(args.model, model, device, test_loader, criterion, epoch=0)

if __name__ == '__main__':
    main()