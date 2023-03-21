import numpy as np
import matplotlib.pyplot as plt

# Prepare data
def generate_linear(n=100):
    plts = np.random.uniform(0, 1, (n,2))
    inputs = []
    labels = []
    for pt in plts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy(n=100):
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)

class NeuronLayer:
    def __init__(self, in_channel, out_channel, lr, activation = 'sigmoid', optimizer = 'sgd'):
        self.weight = np.random.normal(0, 1, (in_channel + 1, out_channel))
        self.momentum = np.zeros((in_channel + 1, out_channel))
        self.sum_of_squares_of_gradients = np.zeros((in_channel + 1, out_channel))
        self.forwardGrad = None
        self.backwardGrad = None
        self.lr = lr
        self.activateFunc = activation
        self.optimizerFunc = optimizer

    def forward(self,inputs):
        self.forwardGrad = np.append(inputs, np.ones((inputs.shape[0], 1)), axis=1)
        if self.activateFunc == 'sigmoid':
            self.output = self.sigmoid(np.matmul(self.forwardGrad, self.weight))
        elif self.activateFunc == 'tanh':
            self.output = self.tanh(np.matmul(self.forwardGrad, self.weight))
        elif self.activateFunc == 'relu':
            self.output = self.ReLU(np.matmul(self.forwardGrad, self.weight))
        elif self.activateFunc == 'lrelu':
            self.output = self.LReLU(np.matmul(self.forwardGrad, self.weight))
        else: # Without activation function
            self.output = np.matmul(self.forwardGrad, self.weight)

        return self.output

    def backward(self, derivative):
        if self.activateFunc == 'sigmoid':
            self.backwardGrad = np.multiply(self.derivativeSigmoid(self.output), derivative)
        elif self.activateFunc == 'tanh':
            self.backwardGrad = np.multiply(self.derivativeTanh(self.output), derivative)
        elif self.activateFunc == 'relu':
            self.backwardGrad = np.multiply(self.derivativeReLU(self.output), derivative)
        elif self.activateFunc == 'lrelu':
            self.backwardGrad = np.multiply(self.derivativeLReLU(self.output), derivative)
        else:# Without activation function
            self.backwardGrad = derivative
        return np.matmul(self.backwardGrad, self.weight[:-1].T)

    def update(self):
        grad = np.matmul(self.forwardGrad.T, self.backwardGrad)
        if self.optimizerFunc == 'sgd':
            deltaWeight = -self.lr * grad
        elif self.optimizerFunc == 'momentum':
            self.momentum = 0.9 * self.momentum - self.lr * grad
            deltaWeight = self.momentum
        elif self.optimizerFunc == 'Adagrad':
            self.sum_of_squares_of_gradients += np.square(grad)
            deltaWeight = -self.lr * grad / np.sqrt(self.sum_of_squares_of_gradients + 1e-8)
        self.weight += deltaWeight
        return self.weight
        

    # -----------------------
    #   activation function
    # -----------------------
    # Sigmoid function
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    # derivative sigmoid
    @staticmethod
    def derivativeSigmoid(y):
        return np.multiply(y, 1.0 - y)
    
    # Tanh function
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    # Derivative of Tanh
    @staticmethod
    def derivativeTanh(y):
        return 1 - y ** 2

    # ReLU function
    @staticmethod
    def ReLU(x):
        return np.maximum(0.0, x)

    # derivative ReLU
    @staticmethod
    def derivativeReLU(y):
        return np.heaviside(y, 0.0)
    
    # Leaky ReLU function
    @staticmethod
    def LReLU(x):
        return np.maximum(0.0, x) + 0.01 * np.minimum(0.0, x)

    # derivative ReLU
    @staticmethod
    def derivativeLReLU(y):
        y[y > 0.0] = 1.0
        y[y <= 0.0] = 0.01
        return y

class NeuralNetwork:
    def __init__(self, hiddenLayerNum = 2, neuronNum = 4, lr = 0.001, epoch = 10000, activationType = 'None', optimizerType = 'sgd', dataType = 'Linear'):
        self.epoch = epoch
        self.dataType = dataType
        self.lr = lr
        self.hidden_units = neuronNum
        self.lossList = []
        self.learningEpoch = []
        self.activationType = activationType
        self.optimizerType = optimizerType
        self.updateWeight = []
        # Input Layer
        self.layer = [NeuronLayer(in_channel = 2, out_channel = neuronNum, lr = lr, activation = self.activationType, optimizer = self.optimizerType)]
        # Hidden Layer
        for i in range(hiddenLayerNum - 1):
            self.layer.append(NeuronLayer(in_channel = neuronNum, out_channel = neuronNum, lr = lr, activation = self.activationType, optimizer = self.optimizerType))
        # Output Layer
        self.layer.append(NeuronLayer(in_channel = neuronNum, out_channel = 1, lr = lr, activation = 'sigmoid', optimizer = self.optimizerType))


    def forward(self, inputs):
        for l in self.layer:
            inputs = l.forward(inputs)
        return inputs

    def backward(self, loss):
        for l in reversed(range(len(self.layer))):
            loss = self.layer[l].backward(loss)
    
    def update(self): # Optimizer
        updateWeight = []
        for l in self.layer:
            self.updateWeight.append(l.update())

    # Loss function
    def MSE(self, yHat, y):
        return np.mean((yHat - y) ** 2)

    def MSE_derivative(self, prediction, groundtruth):
        return 2 * (prediction - groundtruth) / len(prediction)

    # train
    def train(self, x, y):
        for e in range(self.epoch):
            prediction = self.forward(x)
            loss = self.MSE(prediction,y)
            self.backward(self.MSE_derivative(prediction, y))
            self.update()
            
            if e % 100 == 0:
                print(f"Epoch = {e}, loss = {loss}")
                self.lossList.append(loss)
                self.learningEpoch.append(e)

            if loss < 0.001:
                break
        
    # Prediction
    def prediction(self, x):
        predict_y = self.forward(x)
        return predict_y
        
    # Compute accuracy
    def accuracy(self, groundTruth, predict):
        correct = 0.0
        total_loss = 0.0
        print(f"---------- Testing ----------")
        for i in range(len(groundTruth)):
            print(f"Iter = {i+1}   Ground truth = {groundTruth[i]} prediction = {predict[i]}")
            total_loss += self.MSE(groundTruth[i], predict[i])
            if groundTruth[i] == np.round(predict[i]):
                correct += 1
        print(f"loss = {total_loss}, accuracy = {100 * (correct / len(groundTruth))}%")

    # Data visualize
    def show_result(self, x, y):
        # Do prediction
        pred_y = self.prediction(x)
        self.accuracy(y, pred_y)

        # plot groun truth
        plt.subplot(1,2,1)
        plt.title('Ground truth', fontsize=18)
        for i in range(x.shape[0]):
            if y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')
        
        # plot predict result
        plt.subplot(1,2,2)
        plt.title('Predict result', fontsize=18)
        for i in range(x.shape[0]):
            if np.round(pred_y[i]) == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')       
        plt.savefig(f'./output/optimizer/prediction_{self.dataType}_{self.optimizerType}.png')
        # Testing result
        

        # plot learning curve
        plt.figure()
        plt.title('Learning curve', fontsize=18)
        plt.plot(self.learningEpoch, self.lossList)
        plt.savefig(f'./output/optimizer/learning_curve_{self.dataType}_{self.optimizerType}.png')



def main():
    # args initialize
    hiddenLayerNum = 2
    neuronNum = 4
    lr = 0.1
    n = 100
    epoch = 100000
    dataType = 'XOR' # 'XOR'
    activationType = 'none' # sigmoid, tanh, relu, lrelu
    optimizerType = 'sgd' # sgd, momentum, Adagrad, Adam

    # Input data
    if dataType == 'Linear':
        x, y = generate_linear(n)
    elif dataType == 'XOR':
        x, y = generate_XOR_easy(n)
    trainSize = int(len(x) * 0.8)
    # Training
    model = NeuralNetwork(hiddenLayerNum, neuronNum, lr, epoch, activationType, optimizerType, dataType)
    model.train(x[:trainSize], y[:trainSize])
    model.show_result(x[trainSize + 1:], y[trainSize + 1:])
    print("Data type = ",dataType)
    print("Activation function =", activationType)
    print("Optimizer type = ",optimizerType)
        

if __name__ == "__main__":
    main()