# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import random
# # import os 
# # import json
# # from sklearn.feature_extraction.text import CountVectorizer
# # from tqdm import tqdm,trange
# # from time import sleep

# # np.random.seed(0)

# # BATCH_SIZE = 50
# # EPOCHS = 2500
# # LOSS = 0
# # ALPHA = 0.1
# # CATEGORIES = {
# #     '1':'communication',
# #     '2':'weather',
# #     '3':'youtube',
# #     '4':'webbrowser',
# #     '5':'music',
# #     '6':'news',
# #     '7':'todo',
# #     '8':'calendar',
# #     '9':'joikes'
# # }
# # # CATEGORIES = ['communication','weather','youtube','webbrowser','music','news','todo','calendar','joikes']
# # learning_rate = 0.1
# # LossArray = []
# # train_data = {}
# # test_data = {}


# # # Read data and setup maps for integer encoding and decoding.
# # ProjectDir = os.getcwd()
# # file = open('Datasets/MarcusDataset.json','r',encoding='utf-8')
# # DataFile = json.load(file)
# # train_data = DataFile['train_dataset']
# # test_data = DataFile['test_dataset']

# # vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
# # vocab_size = len(vocab)
 
# # print('%d unique words found' % vocab_size)
# # # Assign indices to each word.
# # word_to_idx = { w: i for i, w in enumerate(vocab) }
# # idx_to_word = { i: w for i, w in enumerate(vocab) }

# # from sklearn import datasets
# # iris = datasets.load_iris()
# # dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]

# # def sigmoid(x):
# #     return 1 / (1 + np.exp(-x))

# # def deriv_sigmoid(y):
# #     return y * (1 - y)

# # def tanh(x):
# #     return np.tanh(x)

# # def softmax(xs):
# #     # Applies the Softmax Function to the input array.
# #     return np.exp(xs) / sum(np.exp(xs))

# # def deriv_tanh(y):
# #     return 1 - y * y

# # def cross_entropy(PredictedValue,Target):
# #     return -np.log(PredictedValue[0, Target])

# # def MSE(PredictedValue,TargetValue):
# #     Loss = ((TargetValue - PredictedValue) ** 2).mean()
# #     return Loss

# # class NeuralNetwork():
# #     def __init__(self,LENGHT_DATA,*args,**kwargs):
# #         self.Loss = 0
# #         self.LossArray = []
# #         self.Accuracy = 0
# #         self.AccuracyArray = []

# #         self.LENGHT_DATA = LENGHT_DATA
# #         self.INPUT_LAYERS = self.LENGHT_DATA
# #         self.HIDDEN_LAYERS = 128
# #         self.OUTPUT_LAYERS = 10
# #         # Weights
# #         self.whh = np.random.randn(self.HIDDEN_LAYERS, self.HIDDEN_LAYERS) / 1000
# #         self.wxh = np.random.randn(self.HIDDEN_LAYERS, self.INPUT_LAYERS) / 1000
# #         self.why = np.random.randn(self.OUTPUT_LAYERS, self.HIDDEN_LAYERS) / 1000

# #         # Biases
# #         self.bh = np.zeros((self.HIDDEN_LAYERS, 1))
# #         self.by = np.zeros((self.OUTPUT_LAYERS, 1))

# #     def PreprocessingText(self,text):
# #         '''
# #         Возвращает массив унитарных векторов
# #         которые представляют слова в введенной строке текста
# #         - текст является строкой string
# #         - унитарный вектор имеет форму (vocab_size, 1)
# #         '''
        
# #         Input = []
# #         for w in text.split(' '):
# #             v = np.zeros((vocab_size, 1))
# #             v[word_to_idx[w]] = 1
# #             Input.append(v)
# #         return Input

# #     def FeedForward(self,Input):
# #         '''
# #         Выполнение фазы прямого распространения нейронной сети с
# #         использованием введенных данных.
# #         Возврат итоговой выдачи и скрытого состояния.
# #         - Входные данные в массиве однозначного вектора с формой (input_size, 1).
# #         '''
# #         h = np.zeros((self.whh.shape[0], 1))
 
# #         self.last_inputs = Input
# #         self.last_hs = { 0: h }
 
# #         # Выполнение каждого шага нейронной сети RNN
# #         for i, x in enumerate(Input):
# #             h = np.tanh(self.wxh @ x + self.whh @ h + self.bh)
# #             self.last_hs[i + 1] = h
 
# #         # Подсчет вывода
# #         y = self.why @ h + self.by
 
# #         return y, h

# #     def BackwardPropagation(self,d_y):
# #         '''
# #         Выполнение фазы обратного распространения RNN.
# #         - d_y (dL/dy) имеет форму (output_size, 1).
# #         - learn_rate является вещественным числом float.
# #         '''
# #         n = len(self.last_inputs)
 
# #         # Вычисление dL/dWhy и dL/dby.
# #         d_why = d_y @ self.last_hs[n].T
# #         d_by = d_y
 
# #         # Инициализация dL/dWhh, dL/dWxh, и dL/dbh к нулю.
# #         d_whh = np.zeros(self.whh.shape)
# #         d_wxh = np.zeros(self.wxh.shape)
# #         d_bh = np.zeros(self.bh.shape)
 
# #         # Вычисление dL/dh для последнего h.
# #         d_h = self.why.T @ d_y
 
# #         # Обратное распространение во времени.
# #         for t in reversed(range(n)):
# #             # Среднее значение: dL/dh * (1 - h^2)
# #             temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)
 
# #             # dL/db = dL/dh * (1 - h^2)
# #             d_bh += temp
 
# #             # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
# #             d_whh += temp @ self.last_hs[t].T
 
# #             # dL/dWxh = dL/dh * (1 - h^2) * x
# #             d_wxh += temp @ self.last_inputs[t].T
 
# #             # Далее dL/dh = dL/dh * (1 - h^2) * Whh
# #             d_h = self.whh @ temp
 
# #         # Отсекаем, чтобы предотвратить разрыв градиентов.
# #         for d in [d_wxh, d_whh, d_why, d_bh, d_by]:
# #             np.clip(d, -1, 1, out=d)
 
# #         # Обновляем вес и смещение с использованием градиентного спуска.
# #         self.whh -= learning_rate * d_whh
# #         self.wxh -= learning_rate * d_wxh
# #         self.why -= learning_rate * d_why
# #         self.bh -= learning_rate * d_bh
# #         self.by -= learning_rate * d_by

# #     def train(self,data,BackPropagation):
# #         for epoch in range(EPOCHS):
# #             for i in range(len(data) // 50):

# #                 batch_x, batch_y = zip(*data[i*50 : i*50+50])
# #                 x = np.concatenate(batch_x, axis=0)
# #                 y = np.array(batch_y)
# #                 PredictedValue = self.FeedForward(x)
# #                 STG = PredictedValue
# #                 STG[y] -= 1
# #                 # Backward Propagation
# #                 self.BackwardPropagation(STG)
# #         # bar = trange(EPOCHS,leave=True)
# #         # iteration = 0
# #         # for epoch in bar:
# #         #     items = list(data.items())
# #         #     # random.shuffle(items)

# #         #     loss = 0
# #         #     num_correct = 0

# #         #     for x, y in items:
# #         #         Input = self.PreprocessingText(x)
# #         #         Target = int(y)

# #         #         # Forward
# #         #         Output, OutputHiddenLayer = self.FeedForward(Input)
# #         #         PredictedValue = softmax(Output)

# #         #         # Calculate loss / accuracy
# #         #         loss = MSE(Output,Target)
# #         #         num_correct += int(np.argmax(PredictedValue) == Target)
# #         #         if BackPropagation:
# #         #           # Build dL/dy
# #         #           STG = PredictedValue
# #         #           STG[Target] -= 1
# #         #           # Backward Propagation
# #         #           self.BackwardPropagation(STG)
# #         #         train_loss, train_acc = loss / len(data), num_correct / len(data)
# #         #         bar.set_description(f'Epoch: {epoch}/{EPOCHS}; Loss: {train_loss}; Accuracy: {train_acc}')
# #         #         # if iteration % 100 ==  0:
# #         #         #     print(f'Epoch: {epoch}/{EPOCHS}; Loss: {train_loss}; Accuracy: {train_acc}')
# #         #         # iteration += 1
# #         #     LossArray.append(loss)
# #         # print(LossArray)
# #         # plt.plot(EPOCHS,np.array(LossArray))
# #         # plt.show()
# #     def predict(self,data:str):
# #         Input = self.PreprocessingText(str(data))
# #         print('Input')
# #         print(Input)
# #         # Forward
# #         Output, OutputHiddenLayer = self.FeedForward(Input)
# #         PredictedValue = softmax(Output)
# #         print(PredictedValue)
# #         print('Argmax:' + str(np.argmax(Output)))
# #         print(f'{CATEGORIES[np.argmax(Output)] + 1}')


# # network = NeuralNetwork(vocab_size)
# # network.train(dataset,True)


# # import numpy as np
# # import random
# # import matplotlib.pyplot as plt
# # from tqdm import tqdm,trange
# # from time import sleep
# # import os
# # import json
# # import re
# # import nltk
# # from nltk.corpus import stopwords
# # from nltk.tokenize import word_tokenize
# # import string
# # from sklearn.feature_extraction.text import TfidfVectorizer

# # # Параметры обучения
# # EPOCHS = 2500
# # BATCH_SIZE = 50
# # learning_rate = 0.0002
# # tokenize_text = []
# # train_dataset = {}
# # test_dataset = {}

# # def tanh(x):
# #     return np.tanh(x)

# # def softmax(xs):
# #     # Applies the Softmax Function to the input array.
# #     return np.exp(xs) / sum(np.exp(xs))

# # def deriv_tanh(y):
# #     return 1 - tanh(y) * tanh(y)

# # def cross_entropy(PredictedValue,Target):
# #     return -np.log(PredictedValue[0, Target])

# # def MSE(PredictedValue,TargetValue):
# #     Loss = ((TargetValue - PredictedValue) ** 2).mean()
# #     return Loss

# # def CountAccuracy(PredictedValue,Target,LENGHT_DATA):
# #     CorrectPredictions = 0
# #     PredictedValue = np.argmax(PredictedValue)
# #     if PredictedValue == Target:
# #         CorrectPredictions += 1

# #     Accuracy = CorrectPredictions / LENGHT_DATA
# #     return Accuracy

# # class NeuralNetwork:
# #     def __init__(self,LENGHT_DATA):

# #         # Размер датасета
# #         self.LENGHT_DATA = LENGHT_DATA

# #         # Определние входных,скрытых,выходных слоёв
# #         self.INPUT_LAYERS = self.LENGHT_DATA
# #         self.HIDDEN_LAYERS = 128
# #         self.OUTPUT_LAYERS = 10

# #         # Инициализция гиперпараметров
# #         self.ht = 0
# #         self.hPrevious = 0
# #         self.Output = 0
# #         self.Loss = 0
# #         self.LossArray = []
# #         self.Accuracy = 0
# #         self.AccuracyArray = []

# #         # Инициализация весов
# #         self.whh = np.random.randn(self.HIDDEN_LAYERS, self.HIDDEN_LAYERS) / 1000
# #         self.wxh = np.random.randn(self.HIDDEN_LAYERS, self.INPUT_LAYERS) / 1000
# #         self.why = np.random.randn(self.OUTPUT_LAYERS, self.HIDDEN_LAYERS) / 1000

# #         # Инициализация смещений
# #         self.bh = np.zeros((self.HIDDEN_LAYERS, 1))
# #         self.by = np.zeros((self.OUTPUT_LAYERS, 1))

# #     def PreprocessingText(self,text:str):
# #         Input = []
# #         for w in text.split(' '):
# #             v = np.zeros((vocab_size, 1))
# #             v[word_to_idx[w]] = 1
# #             Input.append(v)
# #         return np.squeeze(np.array(Input))
# #         # text = text.lower()
# #         # text = re.sub(r'\d+', '', text)
# #         # translator = str.maketrans('', '', string.punctuation)
# #         # text.translate(translator)
# #         # text = text
# #         # vectorizer = TfidfVectorizer()
# #         # vectorized_text = vectorizer.fit_transform(text.split('\n'))
# #         # vectorized_text = vectorizer.transform(text.split('\n'))
# #         # vectorized_text = vectorized_text.toarray()
# #         # return np.squeeze(np.array(vectorized_text))

# #     def ForwardPropagation(self,Input):
# #         self.Input = Input
# #         self.ht = np.zeros((self.whh.shape[0], 1))
# #         # self.yt = np.zeros((self.why.shape[0], 1))
# #         for x in Input:
# #             print(x)
# #             self.ht = np.tanh(np.dot(self.wxh,x) + np.dot(self.whh,self.hPrevious) + self.bh)
# #             self.hPrevious = self.ht
# #         self.yt = np.dot(self.why,self.ht) + self.by
# #         self.Output = self.yt
# #         return self.Output

# #     def BackwardPropagation(self,PredictedValue,Target):
# #         d_why = np.zeros_like(self.why)
# #         d_whh = np.zeros_like(self.whh)
# #         d_wxh = np.zeros_like(self.wxh)

# #         Error = PredictedValue - Target
# #         print(deriv_tanh(PredictedValue))
# #         OutputGradient = Error * deriv_tanh(PredictedValue)
# #         d_why = OutputGradient * self.ht.T
# #         HiddenGradient = OutputGradient * self.why.T * deriv_tanh(self.ht.T)
# #         d_whh = HiddenGradient * self.ht.T
# #         InputSum = HiddenGradient * self.whh
# #         InputGradient = InputSum * deriv_tanh(self.Input)
# #         d_wxh = InputGradient * self.Input
# #         self.why = self.why - learning_rate * d_why
# #         self.whh = self.whh - learning_rate * d_whh
# #         self.wxh = self.wxh - learning_rate * d_wxh
# #     def train(self,dataset):
# #         progressbar = trange(EPOCHS,leave=True)
# #         for epoch in progressbar:
# #             items = list(dataset.items())
# #             random.shuffle(items)

# #             self.Loss = 0
# #             self.Accuracy = 0

# #             for x, y in items:
# #                 Input = self.PreprocessingText(x)
# #                 Target = int(y)
# #                 PredictedValue = self.ForwardPropagation(Input)
# #                 self.BackwardPropagation(np.argmax(PredictedValue),Target)
# #                 self.Loss = MSE(PredictedValue,Target)
# #                 self.Accuracy = CountAccuracy(PredictedValue,Target,len(items))
# #                 self.LossArray.append(self.Loss)
# #                 self.AccuracyArray.append(self.Accuracy)
    
# #     def predict(self,data:str):
# #         Input = self.Preprocessing(data)
# #         PredictedValue = self.ForwardPropagation(Input)
# #         return np.argmax(PredictedValue)

# # ProjectDir = os.getcwd()
# # file = open('Datasets/MarcusDataset.json','r',encoding='utf-8')
# # DataFile = json.load(file)
# # train_dataset = DataFile['train_dataset']
# # test_dataset = DataFile['test_dataset']

# # vocab = list(set([w for text in train_dataset.keys() for w in text.split(' ')]))
# # vocab_size = len(vocab)
 
# # print('%d unique words found' % vocab_size)
# # # Assign indices to each word.
# # word_to_idx = { w: i for i, w in enumerate(vocab) }
# # idx_to_word = { i: w for i, w in enumerate(vocab) }
# # network = NeuralNetwork(vocab_size)
# # network.train(train_dataset)

# import math
# text = 'абвгдежзийклмнопрстуфхцчшщъыьэюя.,-_'
# n = math.ceil((math.sqrt(len(text)))) # получение размера квадратной матрицы
# text = iter(text)
# data = [[next(text) for _ in range(6)] for i in range(n)]
# print(data)
# for i in range(n):
#     data.append([])
#     for char in text[i * n: (i + 1) * n]:
#         data[-1].append(char)
# 
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from PreprocessingText import PreprocessingDataset
# from rich.progress import track
# import mplcyberpunk

# plt.style.use("cyberpunk")
# EPOCHS = 50000
# learning_rate = 0.0002
# ProjectDir = os.getcwd()
# Preprocessing = PreprocessingDataset()
# CATEGORIES = ['communication','weather','youtube','webbrowser','music','news','todo','calendar','joikes','exit','time','gratitude','stopwatch','off-stopwatch','pause-stopwatch','unpause-stopwatch','off-music','timer','off-timer','pause-timer','unpause-timer','turn-up-music','turn-down-music','pause-music','unpause-music','shutdown','reboot','hibernation']

# class NeuralNetwork:
#     def __init__(self,LENGHT_DATA):
#         self.LENGHT_DATA = LENGHT_DATA
#         self.INPUT_LAYERS = self.LENGHT_DATA
#         self.HIDDEN_LAYERS = self.LENGHT_DATA
#         self.OUTPUT_LAYERS = len(CATEGORIES)
#         self.w1 = np.random.randn(self.INPUT_LAYERS,self.HIDDEN_LAYERS) / 1000#(np.random.rand(self.INPUT_LAYERS, self.HIDDEN_LAYERS) - 0.5) * 2 * np.sqrt(1/self.INPUT_LAYERS)#np.random.normal(0.0, pow(self.INPUT_LAYERS, -0.5), (self.HIDDEN_LAYERS, self.INPUT_LAYERS))
#         self.w2 = np.random.randn(self.HIDDEN_LAYERS,self.OUTPUT_LAYERS) / 1000#(np.random.rand(self.HIDDEN_LAYERS, self.OUTPUT_LAYERS) - 0.5) * 2 * np.sqrt(1/self.HIDDEN_LAYERS)#np.random.normal(0.0, pow(self.HIDDEN_LAYERS, -0.5), (self.OUTPUT_LAYERS, self.HIDDEN_LAYERS))
#         self.b1 = (np.random.rand(1, self.HIDDEN_LAYERS) - 0.5) * 2 * np.sqrt(1/self.INPUT_LAYERS)#np.zeros((self.HIDDEN_LAYERS, 1))
#         self.b2 = (np.random.rand(1, self.OUTPUT_LAYERS) - 0.5) * 2 * np.sqrt(1/self.HIDDEN_LAYERS)#np.zeros((self.OUTPUT_LAYERS, 1))
#         self.LossArray = []
#         self.Loss = 0
#         self.LocalLoss = 0.5

#     def sigmoid(self,x):
#         return 1 / (1 + np.exp(-x))
    
#     def deriv_sigmoid(self,y):
#         return y * (1 - y)
    
#     def relu(self,x):
#         return x * (x > 0)

#     def deriv_relu(self,x):
#         return (x >= 0).astype(float)

#     def Loss(self,y,output):
#         return 1/2 * (y - output) ** 2
    
#     def Loss_deriv(self,y,output):
#         return y - output

#     def MSE(self,PredictedValue,TargetValue):
#         Loss = ((TargetValue - PredictedValue) ** 2).mean()
#         return Loss
    
#     def CrossEntropy(self,PredictedValue,Target):
#         return -np.log(PredictedValue[0, Target])

#     def softmax(self,xs):
#         # Applies the Softmax Function to the input array.
#         return np.exp(xs) / sum(np.exp(xs))

#     def FeedForwardPropagation(self,Input):
#         self.InputLayer = self.sigmoid(np.dot(Input,self.w1) + self.b1)
#         self.OutputLayer = self.sigmoid(np.dot(self.InputLayer,self.w2) + self.b2)
#         self.Output = self.OutputLayer
#         return self.Output

#     def BackwardPropagation(self,Input,Target):
#         d1_w2 = learning_rate *\
#                 self.Loss_deriv(Target,self.Output) * \
#                 self.deriv_sigmoid(self.Output)
#         d2_w2 = d1_w2 * self.InputLayer.reshape(-1,1)
#         d1_w1 = learning_rate * \
#                 self.Loss_deriv(Target,self.Output) * \
#                 self.deriv_sigmoid(self.Output) @ \
#                 self.w2.T * \
#                 self.deriv_sigmoid(self.InputLayer)
#         d2_w1 = np.matrix(d1_w1).T @ np.matrix(Input)
#         self.w1 += d2_w1
#         self.w2 += d2_w2
    
#     def train(self,TrainInput,TrainTarget):
#         for epoch in  track(range(EPOCHS), description='[green]Training model'):
#             self.Error = 0
#             for Input,Target in zip(TrainInput,TrainTarget):
#                 OutputValue = self.FeedForwardPropagation(Input)
#                 self.BackwardPropagation(Input,Target)
#                 self.Error = self.MSE(self.Output,Target)
#                 if float(self.Error) <= self.LocalLoss and np.argmax(self.Output) == Target:
#                     self.LocalLoss = self.Error
#                     # print('Best model')
#                     self.save()
#             self.LossArray.append(self.Error)
#         # График ошибок
#         plt.title('Train Loss')
#         plt.plot(self.LossArray)
#         plt.savefig(os.path.join(ProjectDir,'Graphics','Loss.png'))
#         # plt.show()
    
#     def predict(self,Input):
#         OutputValue = self.FeedForwardPropagation(Input)
#         PredictedValue = np.argmax(OutputValue)
#         print(PredictedValue)
#         return PredictedValue

#     def save(self,PathParametrs = os.path.join(ProjectDir,'Models','Artyom_NeuralAssistant.npz')):
#         np.savez_compressed(PathParametrs, self.w1,self.w2,self.b1,self.b2)

#     def open(self,PathParametrs = os.path.join(ProjectDir,'Models','Artyom_NeuralAssistant.npz')):
#         ParametrsFile = np.load(PathParametrs)
#         for n in range(int(self.HIDDEN_LAYERS)):
#             for i in range(self.HIDDEN_LAYERS):
#                 if (0 <= n) and (n < len(self.w1)):
#                     if (0 <= i) and (i < len(self.w1[n])):
#                         self.w1[n][i] = ParametrsFile['arr_0'][n][i]
#                 if (0 <= n) and (n < len(self.w2)):
#                     if (0 <= i) and (i < len(self.w2[n])):
#                         self.w2[n][i] = ParametrsFile['arr_1'][n][i]
#                 if (0 <= n) and (n < len(self.b1)):
#                     if (0 <= i) and (i < len(self.b1[n])):
#                         self.b1[n][i] = ParametrsFile['arr_2'][n][i]
#                 if (0 <= n) and (n < len(self.b2)):
#                     if (0 <= i) and (i < len(self.b2[n])):
#                         self.b2[n][i] = ParametrsFile['arr_3'][n][i]
#         print('W1')
#         print(self.w1)
#         print('Parametrs W1')
#         print(ParametrsFile['arr_0'])
    
    

# def TestPredict():
#     while True:
#         command = input('>>>')
#         if command == 'exit':
#             break
#         else:
#             Test = [command]
#             Test = Preprocessing.Start(PredictArray=Test,mode = 'predict')
#             Test = Preprocessing.ToMatrix(Test)
#             network = NeuralNetwork(len(Test))
#             network.open()
#             network.predict(Test)

# if __name__ == '__main__':
#     TestPredict()
# import random
# import numpy as np
# import os
# import json
# from PreprocessingText import PreprocessingDataset
# from rich.progress import track

# CATEGORIES = ['communication','weather','youtube','webbrowser','music','news','todo','calendar','joikes','exit','time','gratitude','stopwatch','off-stopwatch','pause-stopwatch','unpause-stopwatch','off-music','timer','off-timer','pause-timer','unpause-timer','turn-up-music','turn-down-music','pause-music','unpause-music','shutdown','reboot','hibernation']
# ProjectDir = os.getcwd()
# file = open('Datasets/ArtyomDataset.json','r',encoding='utf-8')
# DataFile = json.load(file)
# train_data = DataFile['train_dataset']
# test_data = DataFile['test_dataset']
# Preprocessing = PreprocessingDataset()
# TrainInput,TrainTarget = Preprocessing.Start(Dictionary = train_data,mode = 'train')
# TestInput,TestTarget = Preprocessing.Start(Dictionary = test_data,mode = 'test')
# TrainInput = Preprocessing.ToMatrix(TrainInput)
# TrainTarget = Preprocessing.ToNumpyArray(TrainTarget)
# TestInput = Preprocessing.ToMatrix(TestInput)
# TestTarget = Preprocessing.ToNumpyArray(TestTarget)
# INPUT_DIM = 57
# OUT_DIM = 28
# H_DIM = 512

# def relu(t):
#     return np.maximum(t, 0)

# def softmax(t):
#     out = np.exp(t)
#     return out / np.sum(out)

# def softmax_batch(t):
#     out = np.exp(t)
#     return out / np.sum(out, axis=1, keepdims=True)

# def sparse_cross_entropy(z, y):
#     return -np.log(z[0, y])

# def sparse_cross_entropy_batch(z, y):
#     return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))

# def to_full(y, num_classes):
#     y_full = np.zeros((1, num_classes))
#     y_full[0, y] = 1
#     return y_full

# def to_full_batch(y, num_classes):
#     y_full = np.zeros((len(y), num_classes))
#     for j, yj in enumerate(y):
#         y_full[j, yj] = 1
#     return y_full

# def relu_deriv(t):
#     return (t >= 0).astype(float)

# W1 = np.random.rand(INPUT_DIM, H_DIM)
# b1 = np.random.rand(1, H_DIM)
# W2 = np.random.rand(H_DIM, OUT_DIM)
# b2 = np.random.rand(1, OUT_DIM)

# W1 = (W1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
# b1 = (b1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
# W2 = (W2 - 0.5) * 2 * np.sqrt(1/H_DIM)
# b2 = (b2 - 0.5) * 2 * np.sqrt(1/H_DIM)

# ALPHA = 0.0002
# NUM_EPOCHS = 50000
# BATCH_SIZE = 50

# loss_arr = []

# for ep in track(range(NUM_EPOCHS), description='[green]Training model'):
#     # random.shuffle(dataset)
#     # for i in range(len(dataset) // BATCH_SIZE):
#     for Input,Target in zip(TrainInput,TrainTarget):
#         # batch_x, batch_y = zip(*dataset[i*BATCH_SIZE : i*BATCH_SIZE+BATCH_SIZE])
#         x = Input#np.concatenate(batch_x, axis=0)
#         y = Target#np.array(batch_y)
        
#         # Forward
#         t1 = x @ W1 + b1
#         h1 = relu(t1)
#         t2 = h1 @ W2 + b2
#         z = softmax(t2)
#         E = np.sum(sparse_cross_entropy(z, y))

#         # Backward
#         y_full = to_full(y, OUT_DIM)
#         dE_dt2 = z - y
#         print(len(dE_dt2[0]))
#         dE_dW2 = h1.T @ dE_dt2
#         dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
#         dE_dh1 = dE_dt2 @ W2.T
#         dE_dt1 = dE_dh1 * relu_deriv(t1)
#         dE_dW1 = x.T @ dE_dt1
#         dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

#         # Update
#         W1 = W1 - ALPHA * dE_dW1
#         b1 = b1 - ALPHA * dE_db1
#         W2 = W2 - ALPHA * dE_dW2
#         b2 = b2 - ALPHA * dE_db2
#         if E <=0.1:
#             print(E)
#             print(np.argmax(t2))
#         loss_arr.append(E)

# def predict(x):
#     t1 = x @ W1 + b1
#     h1 = relu(t1)
#     t2 = h1 @ W2 + b2
#     z = softmax(t2)
#     return z

# # def calc_accuracy():
# #     correct = 0
# #     for x, y in dataset:
# #         z = predict(x)
# #         y_pred = np.argmax(z)
# #         if y_pred == y:
# #             correct += 1
# #     acc = correct / len(dataset)
# #     return acc

# # accuracy = calc_accuracy()
# # print("Accuracy:", accuracy)

# import matplotlib.pyplot as plt
# plt.plot(loss_arr)
# plt.show()
# while True:
#     command = input('>>>')
#     if command == 'exit':
#         break
#     else:
#         Test = [command]
#         Test = Preprocessing.Start(PredictArray=Test,mode = 'predict')
#         Test = Preprocessing.ToMatrix(Test)
#         INPUT_DIM = len(Test)
#         W1 = np.random.rand(INPUT_DIM, H_DIM)
#         b1 = np.random.rand(1, H_DIM)
#         W2 = np.random.rand(H_DIM, OUT_DIM)
#         b2 = np.random.rand(1, OUT_DIM)

#         W1 = (W1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
#         b1 = (b1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
#         W2 = (W2 - 0.5) * 2 * np.sqrt(1/H_DIM)
#         b2 = (b2 - 0.5) * 2 * np.sqrt(1/H_DIM)
#         print(np.argmax(predict(Test)))
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import json
# from PreprocessingText import PreprocessingDataset
# from rich.progress import track

# CATEGORIES = ['communication','weather','youtube','webbrowser','music','news','todo','calendar','joikes','exit','time','gratitude','stopwatch','off-stopwatch','pause-stopwatch','unpause-stopwatch','off-music','timer','off-timer','pause-timer','unpause-timer','turn-up-music','turn-down-music','pause-music','unpause-music','shutdown','reboot','hibernation']
# ProjectDir = os.getcwd()
# file = open('Datasets/ArtyomDataset.json','r',encoding='utf-8')
# DataFile = json.load(file)
# train_data = DataFile['train_dataset']
# test_data = DataFile['test_dataset']
# Preprocessing = PreprocessingDataset()
# TrainInput,TrainTarget = Preprocessing.Start(Dictionary = train_data,mode = 'train')
# TestInput,TestTarget = Preprocessing.Start(Dictionary = test_data,mode = 'test')
# TrainInput = Preprocessing.ToMatrix(TrainInput)
# TrainTarget = Preprocessing.ToNumpyArray(TrainTarget)
# TestInput = Preprocessing.ToMatrix(TestInput)
# TestTarget = Preprocessing.ToNumpyArray(TestTarget)

# class NeuralNetwork():
#     def __init__(self, ):
#         INPUT_DIM = len(TrainInput[0])
#         OUT_DIM = len(CATEGORIES)
#         H_DIM = 57

#         self.W1 = np.random.rand(INPUT_DIM, H_DIM)
#         self.b1 = np.random.rand(1, H_DIM)
#         self.W2 = np.random.rand(H_DIM, OUT_DIM)
#         self.b2 = np.random.rand(1, OUT_DIM)

#         self.W1 = (self.W1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
#         self.b1 = (self.b1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
#         self.W2 = (self.W2 - 0.5) * 2 * np.sqrt(1/H_DIM)
#         self.b2 = (self.b2 - 0.5) * 2 * np.sqrt(1/H_DIM)
#         self.limit = 0.5

#         self.error_list = []

#     def forward(self, X):
#         self.z = np.matmul(X, self.W1)
#         self.z2 = self.sigmoid(self.z)
#         self.z3 = np.matmul(self.z2, self.W2)
#         o = self.sigmoid(self.z3)
#         return o

#     def sigmoid(self, s):
#         return 1 / (1 + np.exp(-s))

#     def sigmoidPrime(self, s):
#         return s * (1 - s)

#     def backward(self, X, y, o):
#         self.o_error = y - o
#         self.o_delta = self.o_error * self.sigmoidPrime(o)
#         self.z2_error = np.matmul(self.o_delta, np.matrix.transpose(self.W2))
#         self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
#         self.W1 += np.matmul(np.matrix.transpose(X), self.z2_delta)
#         self.W2 += np.matmul(np.matrix.transpose(self.z2), self.o_delta)

#     def train(self, TrainInput, TrainTarget, epochs):
#         for epoch in range(epochs):
#             for X,y in zip(TrainInput,TrainTarget):
#                 o = self.forward(X)
#                 self.backward(X, y, o)
#                 self.error_list.append(np.abs(self.o_error).mean())
#                 print(np.abs(self.o_error).mean())
#     def predict(self, x_predicted):
#         return self.forward(x_predicted).item()

# network = NeuralNetwork()
# network.train(TrainInput,TrainTarget,10000)
import random
import numpy as np
import nltk 
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import os
import json
from PreprocessingText import PreprocessingDataset

# Подготовка датасета
ProjectDir = os.getcwd()
file = open('Datasets/ArtyomDataset.json','r',encoding='utf-8')
Preprocessing = PreprocessingDataset()
DataFile = json.load(file)
dataset = DataFile['dataset']
TrainInput,TrainTarget = Preprocessing.Start(Dictionary = dataset,mode = 'train')
file.close()

file = open('Settings.json','r',encoding='utf-8')
DataFile = json.load(file)
CATEGORIES = DataFile['CATEGORIES']
CATEGORIES_TARGET = DataFile['CATEGORIES_TARGET']
file.close()

INPUT_DIM = 60
OUT_DIM = 28
H_DIM = 512

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
NUM_EPOCHS = 10000
BATCH_SIZE = 50

loss_arr = []
class NeuralNetwork:
  def __init__(self):
    pass

  def FeedForward(self,x,y):
        global W1
        global b1
        global W2
        global b2
        # Forward
        self.t1 = x @ W1 + b1
        self.h1 = relu(self.t1)
        self.t2 = self.h1 @ W2 + b2
        self.z = softmax_batch(self.t2)
        E = np.sum(sparse_cross_entropy_batch(self.z, y))
        loss_arr.append(E)
        return self.z

  def BackwardPropagation(self,x,y):
        global W1
        global b1
        global W2
        global b2
        # Backward
        y_full = to_full_batch(y, OUT_DIM)
        dE_dt2 = self.z - y_full
        dE_dW2 = self.h1.T @ dE_dt2
        dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_deriv(self.t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

        # Update
        W1 = W1 - ALPHA * dE_dW1
        b1 = b1 - ALPHA * dE_db1
        W2 = W2 - ALPHA * dE_dW2
        b2 = b2 - ALPHA * dE_db2
  def train(self,TrainInput,TrainTarget):
        for ep in range(NUM_EPOCHS):
            # random.shuffle(dataset)
            # for i in range(len(dataset) // BATCH_SIZE):
            # for TrainInput,TrainTarget in zip(Tr,y):
                # batch_x, batch_y = zip(*dataset[i*BATCH_SIZE : i*BATCH_SIZE+BATCH_SIZE])
                
                x = TrainInput#np.concatenate(batch_x, axis=0)
                y = TrainTarget#np.array(batch_y)
                PredictedValue = self.FeedForward(x,y)
                self.BackwardPropagation(x,y)
  def predict(self,x):
      t1 = x @ W1 + b1
      h1 = relu(t1)
      t2 = h1 @ W2 + b2
      z = softmax_batch(t2)
      print(np.argmax(z))
      return z

network = NeuralNetwork()
network.train(TrainInput,TrainTarget)
# network.predict(Preprocessing.Start(PredictArray = ['включи музыку'],mode = 'predict'))
def calc_accuracy():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct / len(dataset)
    return acc

# accuracy = calc_accuracy()
# print("Accuracy:", accuracy)

import matplotlib.pyplot as plt
plt.plot(loss_arr)
plt.show()