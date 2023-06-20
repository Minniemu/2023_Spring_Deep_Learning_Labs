import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor, device, cuda, no_grad
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as op

class EEGNet(nn.Module):
    def __init__(self, activation) -> None:
        super().__init__()

        # Firstconv
        self.firstconv = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=16, 
                kernel_size=(1, 51),
                stride=(1,1),
                padding=(0, 25),
                bias=False
            ), 
            nn.BatchNorm2d(16)
        )

        # depthwiseConv
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, 
                kernel_size=(2, 1),
                stride=(1, 1),
                groups=16,
                bias=False
            ), 
            nn.BatchNorm2d(32),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=0.25)
        )
        
        # seperableConv
        self.seperableConv = nn.Sequential(
            nn.Conv2d(
                in_channels=32, 
                out_channels=32, 
                kernel_size=(1, 15),
                stride=(1, 1),
                padding=(0, 7),
                bias=False
            ), 
            nn.BatchNorm2d(32),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )

        # classify
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=736, out_features=2)
        )

    def forward(self, x):
        # Firstconv
        first_conv = self.firstconv(x)
        depth_wise_conv = self.depthwiseConv(first_conv)
        seperable_conv = self.seperableConv(depth_wise_conv)
        return self.classify(seperable_conv)

class DeepConvNet(nn.Module):
    def __init__(self, activation) -> None:
        super().__init__()
        # Parameters
        out_channels = [25, 25, 50, 100, 200]
        kernel_sizes = [(2, 1),(1, 5),(1, 5),(1, 5)]

        # DeepconvNet 
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channels[0],
                kernel_size=(1,5),
                bias=False
            )
        )
        self.convs = nn.ModuleList()

        for idx in range(4):
            conv_i = nn.Sequential(
                nn.Conv2d(out_channels[idx], out_channels[idx + 1], kernel_size=kernel_sizes[idx]), 
                nn.BatchNorm2d(out_channels[idx + 1]), 
                activation(), 
                nn.MaxPool2d(kernel_size=(1, 2)), 
                nn.Dropout(p=0.5)
            )
            self.convs.append(conv_i)
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8600, 2)
        )
    def forward(self, x):
        x = self.conv0(x)
        for conv_i in self.convs:
            x = conv_i(x)
        return self.classify(x)


def read_bci_data():
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)


    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))
   

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    return train_data, train_label, test_data, test_label

def check_optimizer_type(input_value):
    if input_value == 'adam':
        return op.Adam
    elif input_value == 'adadelta':
        return op.Adadelta
    elif input_value == 'adagrad':
        return op.Adagrad
    elif input_value == 'adamw':
        return op.AdamW
    elif input_value == 'adamax':
        return op.Adamax

def train(train_dataset, test_dataset, epoch_num, batch_size, learning_rate, model_name, device_name, optimizer):
    # Initialize model
    activation_function = [nn.ELU, nn.ReLU, nn.LeakyReLU]
    activation_name = ['ELU', 'ReLU', 'Leaky ReLU']
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset)
    
        
    accuracy = {
        'train':{
            'ELU':[],
            'Leaky ReLU':[],
            'ReLU':[]
        },
        'test':{
            'ELU':[],
            'Leaky ReLU':[],
            'ReLU':[]
        }
    }
    for i in range (len(activation_function)):
        if model_name == 'EEG':
            model = EEGNet(activation_function[i])
        else:
            model = DeepConvNet(activation_function[i])
        model.to(device_name)

        model_optimizer = optimizer(model.parameters(), learning_rate)

        for epoch in tqdm(range (epoch_num)):
            model.train()
            criterion = nn.CrossEntropyLoss()
            train_sum = 0
            test_sum = 0
            # Train 
            for data, label in train_loader:
                label = label.to(torch.long)
                pred = model.forward(data)
                model_optimizer.zero_grad()
                loss = criterion(pred,label)
                loss.backward()
                model_optimizer.step()
                train_sum += torch.sum((torch.max(pred, dim=1)[1] == label).int()).item()
            accuracy['train'][activation_name[i]].append(train_sum / len(train_dataset))
            # Test
            model.eval()
            with torch.no_grad():
                for data, label in test_loader:
                    label = label.to(torch.long)
                    pred = model.forward(data)
                    test_sum += torch.sum((torch.max(pred, dim=1)[1] == label).int()).item()
                accuracy['test'][activation_name[i]].append(test_sum / len(test_dataset))
        print(f"Best accuracy of training({activation_name[i]})= {max(accuracy['train'][activation_name[i]])}")
        print(f"Best accuracy of testing({activation_name[i]}) = {max(accuracy['test'][activation_name[i]])}")
    return accuracy

# Plot the graph
def show_result(accuracy_data, epochs, type, epoch_num, batch_size, optimizer):
    # create data point
    xpoints = np.array([i for i in range(0, epochs)])
    y_train_elu = np.array(accuracy_data['train']['ELU'])
    y_test_elu = np.array(accuracy_data['test']['ELU'])
    y_train_relu = np.array(accuracy_data['train']['ReLU'])
    y_test_relu = np.array(accuracy_data['test']['ReLU'])
    y_train_lrelu = np.array(accuracy_data['train']['Leaky ReLU'])
    y_test_lrelu = np.array(accuracy_data['test']['Leaky ReLU'])

    # plot line
    plt.plot(xpoints, y_train_elu, label='elu train')
    plt.plot(xpoints, y_test_elu, label='elu test')
    plt.plot(xpoints, y_train_relu, label='relu train')
    plt.plot(xpoints, y_test_relu, label='relu test')
    plt.plot(xpoints, y_train_lrelu, label='lrelu train')
    plt.plot(xpoints, y_test_lrelu, label='lrelu test')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.title(f'Activation function comparison({type})')
    plt.legend()
    plt.savefig(f'./optimizer/result_{type}_epoch_{epoch_num}_batch_{batch_size}_{optimizer}.png')

def main():
    # Define parameters
    epoch = 300
    batch_size = 64
    learning_rate = 1e-2
    net_type = 'EEG' # DeepConv
    optimizer_type = 'adamw' # adam, adamw, adamax, adadelta, adagrad
    optimizer = check_optimizer_type(optimizer_type)
    if torch.cuda.is_available(): 
        device = "cuda:5" 
    else: 
        device = "cpu" 
    device = torch.device(device) 
    print("---------- Model info ---------- ")
    print("Net type = ",net_type)
    print("device name = ",device)
    print("epoch = ",epoch)
    print("batch size = ",batch_size)
    print("learning rate = ",learning_rate)
    print("optimizer = ",optimizer)
    print("---------- Accuracy ----------")
    # Read training data
    train_data, train_label, test_data, test_label = read_bci_data()
    train_dataset = TensorDataset(Tensor(train_data).to(device), Tensor(train_label).to(device))
    test_dataset = TensorDataset(Tensor(test_data).to(device), Tensor(test_label).to(device))
    accuracy_data = train(train_dataset, test_dataset, epoch, batch_size, learning_rate, net_type,device, optimizer)
    show_result(accuracy_data, epoch, net_type, epoch, batch_size, optimizer_type)

if __name__ == "__main__":
    main()