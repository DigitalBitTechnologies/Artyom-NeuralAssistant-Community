# Импортирование необходмых библиотек в целях подготовки датасета для нейросети
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
import random

# Подготовка датасета
ProjectDir = os.getcwd()
file = open('Datasets/ArtyomDataset.json','r',encoding='utf-8')
DataFile = json.load(file)
dataset = DataFile['dataset']
file.close()

file = open('Settings.json','r',encoding='utf-8')
DataFile = json.load(file)
CATEGORIES = DataFile['CATEGORIES']
CATEGORIES_TARGET = DataFile['CATEGORIES_TARGET']
file.close()


class PreprocessingDataset:
    def __init__(self):
        self.Dictionary = {}
        self.TrainInput = []
        self.TrainTarget = []
        self.TestInput = []
        self.TestTarget = []
        self.PredictInput = []
        self.PredictArray = []
        self.Mode = 'train'
        self.x = []
        self.y = []

    def ToMatrix(self,array):
        return np.squeeze(array)

    def ToNumpyArray(self,array):
        return np.array(array)

    def Start(self,PredictArray:list = [],Dictionary:dict = {},mode = 'train'):
        self.Mode = mode
        if self.Mode == 'train' or self.Mode == 'test':
            self.Dictionary = list(Dictionary.items())
            random.shuffle(self.Dictionary)
            self.Dictionary = dict(self.Dictionary)
            for intent in self.Dictionary:
                for questions in Dictionary[intent]['questions']:
                    self.x.append(questions)
                    self.y.append(intent)
            if self.Mode == 'train':
                for target in self.y:
                    self.TrainTarget.append(CATEGORIES[target])
            elif self.Mode == 'test':
                for target in self.y:
                    self.TestTarget.append(CATEGORIES[target])
            vectorizer = TfidfVectorizer()
            vectorizer = vectorizer.fit_transform(self.x)
            VectorizedData = vectorizer.toarray()
            InputDatasetFile = open("Datasets/InputDataset.json", "w", encoding ='utf8')
            json.dump(self.x, InputDatasetFile,ensure_ascii=False,sort_keys=True, indent=2)
            InputDatasetFile.close()
            if self.Mode == 'train':
                self.TrainInput = np.squeeze(VectorizedData)
                return self.TrainInput,self.TrainTarget
            elif self.Mode == 'test':
                self.TestInput = VectorizedData
                return self.TestInput,self.TestTarget

        elif self.Mode == 'predict':
            self.PredictArray = PredictArray
            InputDatasetFile = open("Datasets/InputDataset.json", "r", encoding ='utf8')
            DataFile = json.load(InputDatasetFile)
            InputDatasetFile.close()
            vectorizer = TfidfVectorizer()
            vectorizer.fit_transform(DataFile)
            self.PredictInput = np.squeeze(vectorizer.transform(self.PredictArray).toarray())
            return self.PredictInput
