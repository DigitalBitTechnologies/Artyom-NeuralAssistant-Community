import random
import numpy as np
import os
import json
from PreprocessingText import PreprocessingDataset
from rich.progress import track
import mplcyberpunk
import matplotlib.pyplot as plt

plt.style.use("cyberpunk")
CATEGORIES = ['communication','weather','youtube','webbrowser','music','news','todo','calendar','joikes','exit','time','gratitude','stopwatch','off-stopwatch','pause-stopwatch','unpause-stopwatch','off-music','timer','off-timer','pause-timer','unpause-timer','turn-up-music','turn-down-music','pause-music','unpause-music','shutdown','reboot','hibernation']
ProjectDir = os.getcwd()
file = open('Datasets/ArtyomDataset.json','r',encoding='utf-8')
DataFile = json.load(file)
train_data = DataFile['train_dataset']
test_data = DataFile['test_dataset']
Preprocessing = PreprocessingDataset()
TrainInput,TrainTarget = Preprocessing.Start(Dictionary = train_data,mode = 'train')
TestInput,TestTarget = Preprocessing.Start(Dictionary = test_data,mode = 'test')
TrainInput = Preprocessing.ToMatrix(TrainInput)
TrainTarget = Preprocessing.ToNumpyArray(TrainTarget)
TestInput = Preprocessing.ToMatrix(TestInput)
TestTarget = Preprocessing.ToNumpyArray(TestTarget)
INPUT_DIM = len(TrainInput[0])
OUT_DIM = len(CATEGORIES)
H_DIM = 128

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def deriv_sigmoid(y):
    return y * (1 - y)

def relu(t):
    return np.maximum(t, 0)

def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)

def softmax_batch(t):
    out = np.exp(t)
    return out / np.sum(out, axis=1, keepdims=True)

def sparse_cross_entropy(z, y):
    return -np.log(z[0, y])

def sparse_cross_entropy_batch(z, y):
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))

def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full
def MSE(PredictedValue,TargetValue):
        Loss = ((TargetValue - PredictedValue) ** 2).mean()
        return Loss
def to_full_batch(y, num_classes):
    y_full = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full

def relu_deriv(t):
    return (t >= 0).astype(float)

W1 = np.random.rand(INPUT_DIM, H_DIM)
b1 = np.random.rand(1, H_DIM)
W2 = np.random.rand(H_DIM, OUT_DIM)
b2 = np.random.rand(1, OUT_DIM)

W1 = (W1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
b1 = (b1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
W2 = (W2 - 0.5) * 2 * np.sqrt(1/H_DIM)
b2 = (b2 - 0.5) * 2 * np.sqrt(1/H_DIM)

ALPHA = 0.0002
NUM_EPOCHS = 100000
BATCH_SIZE = 50

loss_arr = []


for ep in track(range(NUM_EPOCHS), description='[green]Training model'):
    # random.shuffle(dataset)
    # for i in range(len(dataset) // BATCH_SIZE):
    # for Input,Target in zip(TrainInput,TrainTarget):
        # batch_x, batch_y = zip(*dataset[i*BATCH_SIZE : i*BATCH_SIZE+BATCH_SIZE])
        x = TrainInput#np.concatenate(TrainInput, axis=0)
        y = TrainTarget
        
        # Forward
        t1 = x @ W1 + b1
        h1 = sigmoid(t1)
        t2 = h1 @ W2 + b2
        z = softmax_batch(t2)
        E = np.sum(sparse_cross_entropy_batch(z, y))

        # Backward
        y_full = to_full_batch(y, OUT_DIM)
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * deriv_sigmoid(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

        # Update
        W1 = W1 - ALPHA * dE_dW1
        b1 = b1 - ALPHA * dE_db1
        W2 = W2 - ALPHA * dE_dW2
        b2 = b2 - ALPHA * dE_db2
        # if E <=10:
        #     print(E)
            # print(np.argmax(t2))
        loss_arr.append(E)

def predict(x):
    t1 = x @ W1 + b1
    h1 = sigmoid(t1)
    t2 = h1 @ W2 + b2
    z = softmax_batch(t2)
    return z

def calc_accuracy():
    correct = 0
    for x, y in zip(TrainInput,TrainTarget):
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct / len(TrainInput)
    return acc

accuracy = calc_accuracy()
print("Accuracy:", accuracy)


plt.plot(loss_arr)
plt.savefig(os.path.join(ProjectDir,'Graphics','Loss.png'))
plt.show()
while True:
    command = input('>>>')
    if command == 'exit':
        break
    else:
        Test = [command]
        Test = Preprocessing.Start(PredictArray=Test,mode = 'predict')
        Test = Preprocessing.ToNumpyArray(Test[0])
        print(type(Test))
        print(Test)
        INPUT_DIM = Test.size
        W1 = np.random.rand(INPUT_DIM, H_DIM)
        b1 = np.random.rand(1, H_DIM)
        # W2 = np.random.rand(H_DIM, OUT_DIM)
        # b2 = np.random.rand(1, OUT_DIM)

        W1 = (W1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
        b1 = (b1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
        # W2 = (W2 - 0.5) * 2 * np.sqrt(1/H_DIM)
        # b2 = (b2 - 0.5) * 2 * np.sqrt(1/H_DIM)
        print(np.argmax(predict(Test[0])))