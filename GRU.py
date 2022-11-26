import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os 
import json
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm,trange
from time import sleep
from FirstNeuralNetwork import Output
from PreprocessingText import Preprocessing

np.random.seed(1)

BATCH_SIZE = 50
EPOCHS = 10000
LOSS = 0
ALPHA = 0.1
CATEGORIES = ['communication','weather','youtube','webbrowser','music','news','todo','calendar','joikes','MarcusAssistant']
learning_rate = 0.1
data = []
train_data = {}
test_data = {}


# Read data and setup maps for integer encoding and decoding.
ProjectDir = os.getcwd()
file = open('Datasets/ArtyomDataset.json','r',encoding='utf-8')
DataFile = json.load(file)
train_data = DataFile['train_dataset']
test_data = DataFile['test_dataset']

vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
 
print('%d unique words found' % vocab_size)
# Assign indices to each word.
word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }
# file = open('RuBQ/RuBQ_2.0/RuBQ_2.0_test.json','r',encoding = 'utf-8')
# DataFile = json.load(file)
# for data in DataFile:
#     train_dataset.append(data['question_text'])
#     test_dataset.append(data['answer_text'])
# dataset = open(os.path.join(ProjectDir,'RuBQ/RuBQ_2.0/RuBQ_2.0_paragraphs.json'), 'r',encoding = 'utf-8')
# dataset = json.load(dataset)
# data = []
# for i in dataset:
#     data.append(i['text'])
# list(data)
# data = data[:1000]
# chars = sorted(list(set(data))) # Sort makes model predictable (if seeded).
# data_size, vocab_size = len(data), len(chars)
# print('data has %d characters, %d unique.' % (data_size, vocab_size))
# char_to_ix = { ch:i for i,ch in enumerate(chars) }
# ix_to_char = { i:ch for i,ch in enumerate(chars) }

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def softmax(xs):
    # Applies the Softmax Function to the input array.
    return np.exp(xs) / sum(np.exp(xs))

def deriv_tanh(y):
    return 1 - y * y

def cross_entropy(PredictedValue,Target):
    return -np.log(PredictedValue[0, Target])

def MSE(PredictedValue,TargetValue):
    Loss = ((TargetValue - PredictedValue) ** 2).mean()
    return Loss

class NeuralNetwork():
    def __init__(self,LENGHT_DATA,*args,**kwargs):
        self.Loss = 0
        self.LossArray = []
        self.OutputGradient = 0
        self.Output = 0
        self.OutputPrevious = 0
        self.LENGHT_DATA = LENGHT_DATA
        self.INPUT_LAYER = self.LENGHT_DATA
        self.HIDDEN_LAYER = int(LENGHT_DATA / 2)
        self.OUTPUT_LAYER = 10
        self.InitializationParametrs()
    def InitializationParametrs(self):
        # Count lenght of data
        # self.data = self.Input
        # self.chars = self.data
        # self.LENGHT_DATA = self.Input.size

        # Initialization INPUT_LAYERS,HIDDEN_LAYERS,OUTPUT_LAYERS

        # Initialization weights
        self.wz = np.random.randn(self.HIDDEN_LAYER, self.HIDDEN_LAYER) / 1000
        self.wr = np.random.randn(self.HIDDEN_LAYER, self.HIDDEN_LAYER) / 1000
        self.wh = np.random.randn(self.HIDDEN_LAYER, self.HIDDEN_LAYER) / 1000
        self.wht = np.random.randn(self.OUTPUT_LAYER, self.HIDDEN_LAYER) / 1000

        # self.wz = (np.random.rand(self.INPUT_LAYER, self.HIDDEN_LAYER) - 0.5) * 2 * np.sqrt(1/self.INPUT_LAYER)
        # self.wr = (np.random.rand(self.INPUT_LAYER, self.HIDDEN_LAYER) - 0.5) * 2 * np.sqrt(1/self.INPUT_LAYER)
        # self.wh = (np.random.rand(self.INPUT_LAYER, self.HIDDEN_LAYER) - 0.5) * 2 * np.sqrt(1/self.INPUT_LAYER)
        # self.wht = (np.random.rand(self.HIDDEN_LAYER, self.OUTPUT_LAYER) - 0.5) * 2 * np.sqrt(1/self.HIDDEN_LAYER)

        self.uz = np.random.randn(self.HIDDEN_LAYER, self.HIDDEN_LAYER) / 1000
        self.ur = np.random.randn(self.HIDDEN_LAYER, self.HIDDEN_LAYER) / 1000
        self.uh = np.random.randn(self.HIDDEN_LAYER, self.HIDDEN_LAYER) / 1000

        # self.uz = (np.random.rand(1, self.HIDDEN_LAYER) - 0.5) * 2 * np.sqrt(1/self.INPUT_LAYER)
        # self.ur = (np.random.rand(1, self.HIDDEN_LAYER) - 0.5) * 2 * np.sqrt(1/self.INPUT_LAYER)
        # self.uh = (np.random.rand(1, self.HIDDEN_LAYER) - 0.5) * 2 * np.sqrt(1/self.INPUT_LAYER)

        # Initialization biases
        self.bz = np.zeros((self.HIDDEN_LAYER, 1))
        self.br = np.zeros((self.HIDDEN_LAYER, 1))
        self.bh = np.zeros((self.HIDDEN_LAYER, 1))
        self.bht = np.zeros((self.OUTPUT_LAYER, 1))

    

    def PreprocessingDataset(self,text):
        '''
        Возвращает массив унитарных векторов
        которые представляют слова в введенной строке текста
        - текст является строкой string
        - унитарный вектор имеет форму (vocab_size, 1)
        '''
        
        Input = []
        for w in text.split(' '):
            v = np.zeros((vocab_size, 1))
            v[word_to_idx[w]] = 1
            Input.append(v)
        return Input

    def FeedFoward(self):
        self.OutputPrevious = self.Output
        # self.rt = sigmoid(np.dot(self.wr,self.OutputPrevious) + np.dot(self.ur,self.Input))
        # self.zt = sigmoid(np.dot(self.wz,self.OutputPrevious) + np.dot(self.uz,self.Input))
        # self.h = tanh(np.dot((self.rt * self.OutputPrevious),self.wh) + np.dot(self.uh,self.Input))
        # self.ht = (1 - self.zt) * self.h + self.zt * self.OutputPrevious
        # self.Output = self.ht
        # print("Output:" + str(sum(softmax(self.Output))))
        # return self.Output

        self.zt = sigmoid(np.dot(self.wz,self.Input) + np.dot(self.uz,self.OutputPrevious) +  self.bz)#sigmoid(self.wz * np.dot(self.OutputPrevious,self.Input))
        self.rt = sigmoid(np.dot(self.wr,self.Input) + np.dot(self.ur,self.OutputPrevious) +  self.br)#sigmoid(self.wr * np.dot(self.OutputPrevious,self.Input))
        self.ht = tanh(np.dot(self.wh,self.Input) + np.dot(self.uh,self.OutputPrevious) +  self.bh)#tanh(self.wht * np.dot(self.rt * self.OutputPrevious,self.Input))
        self.Output = np.multiply(self.zt,self.OutputPrevious) + np.multiply((1 - self.zt),self.ht)#(1 - self.zt) * self.OutputPrevious + self.zt * self.ht
        print("Output:" + str(sum(softmax(self.Output))))
        return self.Output
    
    def BackwardPropagation(self):
        # Инициализация параметров градиента
        # Веса
        self.dwz = np.zeros_like(self.wz)
        self.dwr = np.zeros_like(self.wr)
        self.dwh = np.zeros_like(self.wh)
        self.dwht = np.zeros_like(self.wht)

        self.duz = np.zeros_like(self.uz)
        self.dur = np.zeros_like(self.ur)
        self.duh = np.zeros_like(self.uh)

        # Пороги вхождения
        self.dbz = np.zeros_like(self.bz)
        self.dbr = np.zeros_like(self.br)
        self.dbh = np.zeros_like(self.bh)
        self.dbht = np.zeros_like(self.bht)

        # Расчёт градиента скрытого слоя
        self.d0 = self.OutputGradient
        self.d1 = self.zt * self.d0
        self.d2 = self.OutputPrevious * self.d0
        self.d3 = self.h * self.d0
        self.d4 = -1 * self.d3
        self.d5 = self.d2 + self.d4
        self.d6 = (1 - self.zt) * self.d0
        self.d7 = self.d5 * (self.zt * (1 - self.zt))
        self.d8 = self.d6 * (1 - self.h)
        self.d9 = np.dot(self.d8,self.uh.T)
        self.d10 = np.dot(self.d8,self.wh.T)
        self.d11 = np.dot(self.d7,self.uz.T)
        self.d12 = np.dot(self.d7,self.wz.T)
        self.d14 = self.d10 * self.rt
        self.d15 = self.d10 * self.OutputPrevious
        self.d16 = self.d15 * (self.rt * (1 - self.rt))
        self.d13 = np.dot(self.d16,self.ur.T)
        self.d17 = np.dot(self.d16,self.wr.T)

        # Расчёт градиента входного слоя
        self.dOutput = self.d9 + self.d11 + self.d13
        self.dOutputPrevious = self.d12 + self.d14 + self.d1 + self.d17
        self.dur = np.dot(self.Input.T,self.d16)
        self.duz = np.dot(self.Input.T,self.d7)
        self.duh = np.dot(self.Input.T,self.d8)
        self.dwr = np.dot(self.OutputPrevious,self.d16)
        self.dwz = np.dot(self.OutputPrevious,self.d7)
        self.dwh = np.dot((self.OutputPrevious * self.rt).T,self.d8)

        # Назначение новых параметров для нейросети
        self.wz = self.dwz
        self.wr = self.dwr
        self.wh = self.dwh
        self.uz = self.duz
        self.ur = self.dur
        self.uh = self.duh 

    def fit(self,data,BackwardPropagation):
        bar = trange(EPOCHS,leave=True)
        for epoch in bar:
            items = list(data.items())
            random.shuffle(items)
            for x,y in items:
                self.LENGHT_DATA = len(train_data)
                self.Input = np.squeeze(self.PreprocessingDataset(x))
                # self.Input.reshape(len(self.Input) / 2,len(self.Input) / 2)
                self.Target = int(y)
                self.Predictedvalue = self.FeedFoward()
                self.BackwardPropagation()
                self.Loss = MSE(self.Predictedvalue,self.Target)
                self.LossArray.append(self.Loss)
            bar.set_description(f'Epoch: {epoch}/{EPOCHS}; Loss: {self.Loss}')
            # Удаление значений из батча по окончанию эпохи
            # for value in train_batch:
            #     train_batch.remove(value)
            sleep(0.1)
            
        plt.plot(self.LossArray)
        plt.show()
    def predict(self,data:str):
        Input = self.PreprocessingText(str(data))
        print('Input')
        print(Input)
        # Forward
        Output = self.FeedForward(Input)
        PredictedValue = softmax(Output)
        print(PredictedValue)
        print('Argmax:' + str(np.argmax(Output)))
        print(f'{CATEGORIES[np.argmax(Output)] + 1}')

    def open(self):
        pass

    def save(self):
        pass

    def Start(self):
        for i in range(EPOCHS):
            PredictedValue = self.FeedFoward()
            BackPropagationValue = self.BackwardPropagation()
        # print(PredictedValue)
            print('Loss:' + str(((self.Target - PredictedValue) ** 2).mean()))

network = NeuralNetwork(vocab_size)
network.fit(train_data,True)
network.fit(train_data,False)

vocab = list(set([w for text in test_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
 
print('%d unique words found' % vocab_size)
# Assign indices to each word.
word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }

network = NeuralNetwork(vocab_size)
network.fit(test_data,True)
network.fit(test_data,False)

vocab = list(set([w for text in ['Как дела?'] for w in text.split(' ')]))
vocab_size = len(vocab)
 
print('%d unique words found' % vocab_size)
# Assign indices to each word.
word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }
network = NeuralNetwork(vocab_size)

network.predict('Как дела?')

vocab = list(set([w for text in ['Музыка'] for w in text.split(' ')]))
vocab_size = len(vocab)
 
print('%d unique words found' % vocab_size)
# Assign indices to each word.
word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }
network = NeuralNetwork(vocab_size)

network.predict('Музыка')