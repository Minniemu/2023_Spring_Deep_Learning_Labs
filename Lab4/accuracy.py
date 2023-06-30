import matplotlib.pyplot as plt
import numpy as np
def parse_file(file_name):
    f = open(file_name, 'r')
    Training = []
    Testing = []
    for line in f.readlines():
        if line.startswith('Train'):
            tmp = line.split(',')[0].split(' ')[-2]
            Training.append(float(tmp))
        elif line.startswith('Test'):
            tmp = line.split(',')[0].split(' ')[-2]
            Testing.append(float(tmp))
    f.close()
    return np.array(Training), np.array(Testing)
# resnet18_train, resnet18_test = parse_file('train_resnet18v5.txt')
# resnet18wo_train, resnet18wo_test = parse_file('train_resnet18wov5.txt')
resnet50_train, resnet50_test = parse_file('train_resnet50v4.txt')
resnet50wo_train, resnet50wo_test = parse_file('train_resnet50wov5.txt')


xpoints = np.array([i for i in range(0, 10)])
# plt.plot(xpoints, resnet18_train, color='blue', label='ResNet18 train')
# plt.plot(xpoints, resnet18_test, color='blue', linestyle="-.",label='ResNet18 test')
# plt.plot(xpoints, resnet18wo_train, color='red', label='ResNet18wo train')
# plt.plot(xpoints, resnet18wo_test, color='red', linestyle="--", label='ResNet18wo test')
# plt.legend()
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy(%)")
# plt.title('Result Comparison(ResNet18)')
# plt.savefig('Resnet18.png')
plt.plot(xpoints, resnet50_train, color='blue', label='ResNet50 train')
plt.plot(xpoints, resnet50_test, color='blue', linestyle="-.",label='ResNet50 test')
plt.plot(xpoints, resnet50wo_train, color='red', label='ResNet50wo train')
plt.plot(xpoints, resnet50wo_test, color='red', linestyle="--", label='ResNet50wo test')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy(%)")
plt.title('Result Comparison(ResNet50)')
plt.savefig('Resnet50.png')