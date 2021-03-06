from tempfile import template

import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from keras.metrics import categorical_crossentropy, categorical_accuracy

from keras.datasets import cifar10

import matplotlib.pyplot as plt

import asyncio
import websockets
import random
import json
import os.path
import math
import time

class Server:
    def __init__(self):
        self.ip_address = 'localhost'  # ip адрес для открытия вебсокета
        self.port = 8771  # Порт для подключения клиентов

        self.client_ID = 1 # ID, которое выдаётся клиенту

        self.train_images = 50000  # Количество изображений для обучения
        self.total_received_tasks = 0 # Общее количество полученных результатов от клиента
        # self.epochs_total = 3  # Общее количество эпох обучения
        self.epochs_computing = 0 # Количество эпох, пройденных на данный момент
        # self.epochs_per_person = 1  # Количество эпох обучения для одного узла
        # self.part_of_images = 0.01  # Доля изображений для хоста для обучение нейронной сети (изображения генерируются случайным образом)

        self.epochs_total = 50  # Общее количество эпох обучения
        self.epochs_per_person = 10  # Количество эпох обучения для одного узла
        self.part_of_images = 0.1  # Доля изображений для хоста для обучение нейронной сети (изображения генерируются случайным образом)

        self.weights_filename_h5 = 'weights.h5' # Файл для хранения начальных значений весов в формате h5
        self.weights_filename_txt = 'weights.txt' # Файл для хранения весов в txt формате
        self.weights_result_filename_h5 = 'weights_result.h5' # Файл для хранения весов после обучения в формате h5
        self.weights = None  # Глобальные значения весов модели

        self.confusion_matrix_filename = 'confusion_matrix.txt' # Файл для хранения confusion matrix

        self.model = None # Модель для тестирования нейронной сети

        # Данные датасета
        self.X_test = []
        self.y_test = []
        self.y_test_base = []
        self.X_train = []
        self.y_train = []
        self.y_train_base = []

        self.class_num = None

        # Параметры для обучения
        self.loss = 'categorical_crossentropy'
        self.optimizer = 'adam'
        self.metrics = ['accuracy']

        self.log_name = 'server_log.txt' # Имя файла для логов
        self.log = open(self.log_name, 'w') # Файл с логами

        # Переменные с метриками
        self.confusion_matrix = []

        self.accuracy = []

        self.precision_macro = []
        self.precision_micro = []
        self.precision_weighted = []

        self.recall_macro = []
        self.recall_micro = []
        self.recall_weighted = []

    # Изменить некоторые начальные значения
    def change_data(self):
        answer = input('Изменить значения по умолчанию? (y|n) ')
        if(answer == 'y'):
            message = 'Введите ip-адрес, на котором будем открывать websocket соединение: '
            self.ip_address = (input(message))
            message = 'Введите номер порта, на котором будем открывать websocket соединение: '
            self.port = int(input(message))
            message = 'Введите общее количество эпох обучения (по умолчанию ' + str(self.epochs_total) + ') '
            self.epochs_total = int(input(message))
            message = 'Введите количество эпох обучения для одного клиента (по умолчанию ' + str(self.epochs_per_person) + ') '
            self.epochs_per_person = int(input(message))
            message = 'Введите долю изображений из обучающей выборки, отправляемых клиенту (по умолчанию ' + str(self.part_of_images) + ') '
            self.part_of_images = float(input(message))

    # Начальное обучение нейронной сети на нескольких эпохах для получения начальных значений весов
    def get_base_weights(self):
        print('Создание модели и тестовое обучение на 1-ой эпохе')
        print()

        self.log.writelines('Создание модели и тестовое обучение на 1-ой эпохе\n')
        self.log.writelines('\n')

        # loading in the data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()

        # normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0

        # one hot encode outputs
        self.y_train = np_utils.to_categorical(self.y_train)
        self.y_test = np_utils.to_categorical(self.y_test)
        self.class_num = self.y_test.shape[1]

        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), input_shape=self.X_train.shape[1:], padding='same'))
        self.model.add(Activation('relu'))

        self.model.add(Dropout(0.2))

        self.model.add(BatchNormalization())

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Dropout(0.2))

        self.model.add(Dense(256, kernel_constraint=maxnorm(3)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(Dense(128, kernel_constraint=maxnorm(3)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())

        self.model.add(Dense(self.class_num))
        self.model.add(Activation('softmax'))

        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        self.model.fit(self.X_train, self.y_train, epochs=1, verbose=1)

        self.model.save_weights(self.weights_filename_h5)
        self.weights = self.model.get_weights()

        message1 = 'Начальные значения весов сохранены в файле ' + self.weights_filename_h5

        print('Модель создана и пройдено тестовое обучение')
        print(message1)
        print('Сервер запущен')
        print()

        self.log.writelines('Модель создана и пройдено тестовое обучение\n')
        self.log.writelines(message1)
        self.log.writelines('\n')
        self.log.writelines('Сервер запущен\n')
        self.log.writelines('\n')

    # Создание модели и загрузка начальных значений весов
    def load_base_weights(self):
        print('Создание модели и загрузка весов')
        print()

        self.log.writelines('Создание модели и загрузка весов\n')
        self.log.writelines('\n')

        # loading in the data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()

        # normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0

        self.y_train_base = self.y_train
        self.y_test_base = self.y_test
        self.y_train_base = np.reshape(self.y_train_base, 50000)
        self.y_test_base = np.reshape(self.y_test_base, 10000)

        # one hot encode outputs
        self.y_train = np_utils.to_categorical(self.y_train)
        self.y_test = np_utils.to_categorical(self.y_test)
        self.class_num = self.y_test.shape[1]

        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), input_shape=self.X_train.shape[1:], padding='same'))
        self.model.add(Activation('relu'))

        self.model.add(Dropout(0.2))

        self.model.add(BatchNormalization())

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Dropout(0.2))

        self.model.add(Dense(256, kernel_constraint=maxnorm(3)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(Dense(128, kernel_constraint=maxnorm(3)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())

        self.model.add(Dense(self.class_num))
        self.model.add(Activation('softmax'))

        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        self.model.load_weights(self.weights_filename_h5)
        self.weights = self.model.get_weights()

        message1 = 'Начальные значения загружены из файла ' + self.weights_filename_h5

        print('Модель создана и веса загружены')
        print(message1)
        print('Сервер запущен')
        print()

        self.log.writelines('Модель создана и веса загружены\n')
        self.log.writelines(message1)
        self.log.writelines('\n')
        self.log.writelines('Сервер запущен\n')
        self.log.writelines('\n')

        # self.test()

    # Метод для изменения глобального значения весов
    def change_global_weights(self, delta_weights):
        print('Корректировка глобальных весов нейронной сети')
        self.log.writelines('Корректировка глобальных весов нейронной сети\n')

        for i2 in range(len(self.weights[0])):
            for i3 in range(len(self.weights[0][i2])):
                for i4 in range(len(self.weights[0][i2][i3])):
                    for i5 in range(len(self.weights[0][i2][i3][i4])):
                        self.weights[0][i2][i3][i4][i5] = self.weights[0][i2][i3][i4][i5] + delta_weights[0][i2][i3][i4][i5]

        for i1 in range(5):
            for i2 in range(len(self.weights[i1 + 1])):
                self.weights[i1 + 1][i2] = self.weights[i1 + 1][i2] + delta_weights[i1 + 1][i2]

        for i2 in range(len(self.weights[6])):
            for i3 in range(len(self.weights[6][i2])):
                for i4 in range(len(self.weights[6][i2][i3])):
                    for i5 in range(len(self.weights[6][i2][i3][i4])):
                        self.weights[6][i2][i3][i4][i5] = self.weights[6][i2][i3][i4][i5] + delta_weights[6][i2][i3][i4][i5]

        for i1 in range(5):
            for i2 in range(len(self.weights[i1 + 7])):
                self.weights[i1 + 7][i2] = self.weights[i1 + 7][i2] + delta_weights[i1 + 7][i2]

        for i2 in range(len(self.weights[12])):
            for i3 in range(len(self.weights[12][i2])):
                self.weights[12][i2][i3] = self.weights[12][i2][i3] + delta_weights[12][i2][i3]

        for i1 in range(5):
            for i2 in range(len(self.weights[i1 + 13])):
                self.weights[i1 + 13][i2] = self.weights[i1 + 13][i2] + delta_weights[i1 + 13][i2]

        for i2 in range(len(self.weights[18])):
            for i3 in range(len(self.weights[18][i2])):
                self.weights[18][i2][i3] = self.weights[18][i2][i3] + delta_weights[18][i2][i3]

        for i1 in range(5):
            for i2 in range(len(self.weights[i1 + 19])):
                self.weights[i1 + 19][i2] = self.weights[i1 + 19][i2] + delta_weights[i1 + 19][i2]

        for i2 in range(len(self.weights[24])):
            for i3 in range(len(self.weights[24][i2])):
                self.weights[24][i2][i3] = self.weights[24][i2][i3] + delta_weights[24][i2][i3]

        for i2 in range(len(self.weights[25])):
            self.weights[25][i2] = self.weights[25][i2] + delta_weights[25][i2]

        print('Глобальные значения весов успешно обновлены')
        self.log.writelines('Глобальные значения весов успешно обновлены\n')

    # Тестирование обученной нейронной сети на тестовых данных
    def test_nn(self):
        print('Выполняется прогнозирование тестовых данных')
        self.log.writelines('Выполняется прогнозирование тестовых данных\n')

        self.model.set_weights(self.weights) # Загрузка глобальных значений весов в модель
        y_predict = self.model.predict(x=self.X_test) # Прогнозирование
        y_predict = np.argmax(y_predict, axis=-1) # Возвращаем обратно метки классов

        print('Прогнозирование тестовых данных успешно выполнено')
        print('Выполняется расчёт метрик на тестовых данных')
        self.log.writelines('Прогнозирование тестовых данных успешно выполнено\n')
        self.log.writelines('Выполняется расчёт метрик на тестовых данных\n')

        # Расчёт метрик
        # 1) Расчёт confusion_matrix
        cm = confusion_matrix(self.y_test_base, y_predict)
        self.confusion_matrix.append(cm)
        message = 'confusion_matrix:\n' + str(cm)
        print(message)
        self.log.writelines(message)
        self.log.writelines('\n')

        # 2) Расчёт метрики accuracy
        accuracy = accuracy_score(self.y_test_base, y_predict)
        self.accuracy.append(accuracy)
        message = 'accuracy: ' + str(round(accuracy, 4))
        print(message)
        self.log.writelines(message)
        self.log.writelines('\n')

        # 3) Расчёт метрики precision (macro, micro, weighted)
        precision_macro = precision_score(self.y_test_base, y_predict, average='macro')
        precision_micro = precision_score(self.y_test_base, y_predict, average='micro')
        precision_weighted = precision_score(self.y_test_base, y_predict, average='weighted')
        self.precision_macro.append(precision_macro)
        self.precision_micro.append(precision_micro)
        self.precision_weighted.append(precision_weighted)
        message1 = 'precision_score (macro): ' + str(round(precision_macro, 4))
        message2 = 'precision_score (micro): ' + str(round(precision_micro, 4))
        message3 = 'precision_score (weighted): ' + str(round(precision_weighted, 4))
        print(message1)
        print(message2)
        print(message3)
        self.log.writelines(message1)
        self.log.writelines('\n')
        self.log.writelines(message2)
        self.log.writelines('\n')
        self.log.writelines(message3)
        self.log.writelines('\n')

        # 4) Расчёт метрики recall (macro, micro, weighted)
        recall_macro = recall_score(self.y_test_base, y_predict, average='macro')
        recall_micro = recall_score(self.y_test_base, y_predict, average='micro')
        recall_weighted = recall_score(self.y_test_base, y_predict, average='weighted')
        self.recall_macro.append(recall_macro)
        self.recall_micro.append(recall_micro)
        self.recall_weighted.append(recall_weighted)
        message1 = 'recall_score (macro): ' + str(round(recall_macro, 4))
        message2 = 'recall_score (micro): ' + str(round(recall_micro, 4))
        message3 = 'recall_score (weighted): ' + str(round(recall_weighted, 4))
        print(message1)
        print(message2)
        print(message3)
        self.log.writelines(message1)
        self.log.writelines('\n')
        self.log.writelines(message2)
        self.log.writelines('\n')
        self.log.writelines(message3)
        self.log.writelines('\n')

        print('Расчёт метрик на тестовых данных успешно произведён')
        print()
        self.log.writelines('Расчёт метрик на тестовых данных успешно произведён\n')
        self.log.writelines('\n')

        return cm

    # Метод для отправки весов
    async def send_weights(self, websocket):
        start_time = time.time() # Задаём счётчик времени
        print('Идёт отправка весов. Завершено на 0%')
        self.log.writelines('Идёт отправка весов. Завершено на 0%\n')
        total_weights = 4250058 # Общее количество весов нейронной сети
        send_weights = 0 # Количество отправленных весов
        send_persentage = 0 # Процент отправленных весов

        for i2 in range(len(self.weights[0])):
            for i3 in range(len(self.weights[0][i2])):
                for i4 in range(len(self.weights[0][i2][i3])):
                    for i5 in range(len(self.weights[0][i2][i3][i4])):
                        await websocket.send(str(self.weights[0][i2][i3][i4][i5]))
                        send_weights = send_weights + 1

        for i1 in range(5):
            for i2 in range(len(self.weights[i1 + 1])):
                await websocket.send(str(self.weights[i1 + 1][i2]))
                send_weights = send_weights + 1

        for i2 in range(len(self.weights[6])):
            for i3 in range(len(self.weights[6][i2])):
                for i4 in range(len(self.weights[6][i2][i3])):
                    for i5 in range(len(self.weights[6][i2][i3][i4])):
                        await websocket.send(str(self.weights[6][i2][i3][i4][i5]))
                        send_weights = send_weights + 1

        for i1 in range(5):
            for i2 in range(len(self.weights[i1 + 7])):
                await websocket.send(str(self.weights[i1 + 7][i2]))
                send_weights = send_weights + 1

        for i2 in range(len(self.weights[12])):
            for i3 in range(len(self.weights[12][i2])):
                await websocket.send(str(self.weights[12][i2][i3]))
                send_weights = send_weights + 1
                if(send_persentage < math.floor(send_weights / total_weights * 100)):
                    send_persentage = send_persentage + 1
                    message = 'Идёт отправка весов клиенту. Завершено на ' + str(send_persentage) + '%'
                    print(message)
                    self.log.writelines(message)
                    self.log.writelines('\n')

        for i1 in range(5):
            for i2 in range(len(self.weights[i1 + 13])):
                await websocket.send(str(self.weights[i1 + 13][i2]))
                send_weights = send_weights + 1

        for i2 in range(len(self.weights[18])):
            for i3 in range(len(self.weights[18][i2])):
                await websocket.send(str(self.weights[18][i2][i3]))
                send_weights = send_weights + 1

        for i1 in range(5):
            for i2 in range(len(self.weights[i1 + 19])):
                await websocket.send(str(self.weights[i1 + 19][i2]))
                send_weights = send_weights + 1

        for i2 in range(len(self.weights[24])):
            for i3 in range(len(self.weights[24][i2])):
                await websocket.send(str(self.weights[24][i2][i3]))
                send_weights = send_weights + 1

        for i2 in range(len(self.weights[25])):
            await websocket.send(str(self.weights[25][i2]))
            send_weights = send_weights + 1

        end_time = time.time()
        message = 'Время отправки весов составило ' + str(round((end_time - start_time) / 60, 2)) + ' минут'

        print('Веса успешно отправлены')
        print(message)
        print()
        self.log.writelines('Веса успешно отправлены\n')
        self.log.writelines(message)
        self.log.writelines('\n')
        self.log.writelines('\n')

    # Метод для приёма весов
    async def get_weights(self, websocket):
        start_time = time.time() # Задаём счётчик времени
        print('Загрузка весов от клиента... Завершено на 0%')
        self.log.writelines('Загрузка весов от клиента... Завершено на 0%\n')
        total_weights = 4250058 # Общее количество весов нейронной сети
        get_weights = 0 # Количество отправленных весов
        get_persentage = 0 # Процент отправленных весов
        weights = []

        weights0 = []
        for i2 in range(3):
            weights1 = []
            for i3 in range(3):
                weights2 = []
                for i4 in range(3):
                    weights3 = []
                    for i5 in range(32):
                        weights3.append(float(await websocket.recv()))
                        get_weights = get_weights + 1
                    weights2.append(np.array(weights3))
                weights1.append(np.array(weights2))
            weights0.append(np.array(weights1))
        weights.append(np.array(weights0))

        for i1 in range(5):
            weights1 = []
            for i2 in range(32):
                weights1.append(float(await websocket.recv()))
                get_weights = get_weights + 1
            weights.append(np.array(weights1))

        weights0 = []
        for i2 in range(3):
            weights1 = []
            for i3 in range(3):
                weights2 = []
                for i4 in range(32):
                    weights3 = []
                    for i5 in range(64):
                        weights3.append(float(await websocket.recv()))
                        get_weights = get_weights + 1
                    weights2.append(np.array(weights3))
                weights1.append(np.array(weights2))
            weights0.append(np.array(weights1))
        weights.append(np.array(weights0))

        for i1 in range(5):
            weights1 = []
            for i2 in range(64):
                weights1.append(float(await websocket.recv()))
                get_weights = get_weights + 1
            weights.append(np.array(weights1))

        weights0 = []
        for i2 in range(16384):
            weights1 = []
            for i3 in range(256):
                weights1.append(float(await websocket.recv()))
                get_weights = get_weights + 1
                if(get_persentage < math.floor(get_weights / total_weights * 100)):
                    get_persentage = get_persentage + 1
                    message = 'Идёт загрузка весов от клиента. Завершено на ' + str(get_persentage) + '%'
                    print(message)
                    self.log.writelines(message)
                    self.log.writelines('\n')
            weights0.append(np.array(weights1))
        weights.append(np.array(weights0))

        for i1 in range(5):
            weights1 = []
            for i2 in range(256):
                weights1.append(float(await websocket.recv()))
                get_weights = get_weights + 1
            weights.append(np.array(weights1))

        weights0 = []
        for i2 in range(256):
            weights1 = []
            for i3 in range(128):
                weights1.append(float(await websocket.recv()))
                get_weights = get_weights + 1
            weights0.append(np.array(weights1))
        weights.append(np.array(weights0))

        for i1 in range(5):
            weights1 = []
            for i2 in range(128):
                weights1.append(float(await websocket.recv()))
                get_weights = get_weights + 1
            weights.append(np.array(weights1))

        weights0 = []
        for i2 in range(128):
            weights1 = []
            for i3 in range(10):
                weights1.append(float(await websocket.recv()))
                get_weights = get_weights + 1
            weights0.append(np.array(weights1))
        weights.append(np.array(weights0))

        weights0 = []
        for i2 in range(10):
            weights0.append(float(await websocket.recv()))
            get_weights = get_weights + 1
        weights.append(np.array(weights0))

        end_time = time.time()
        message = 'Время загрузки весов составило ' + str(round((end_time - start_time) / 60, 2)) + ' минут'

        print('Веса успешно загружены')
        print(message)
        print()
        self.log.writelines('Веса успешно загружены\n')
        self.log.writelines(message)
        self.log.writelines('\n')
        self.log.writelines('\n')

        return weights

    # websocket метод для прослушивания заданного порта и реакции на запросы
    async def data_exchange(self, websocket):
        message = await websocket.recv() # Приём первоначального блока данных

        if(message == 'GET TASK'): # Если клиент запросил "задание"
            if(self.epochs_total != 0): # Если не все "задания" ещё выданы
                print('Клиент запросил задание с сервера')
                print('Выполняется генерирование задания...')
                self.log.writelines('Клиент запросил задание с сервера\n')
                self.log.writelines('Выполняется генерирование задания...\n')

                # Сгенерировать случайным образом изображения для обучения
                rand_images = []
                l = list(range(1, 50001))
                random.shuffle(l)
                for i in range(round(self.part_of_images * self.train_images)):
                    rand_images.append(l[i])

                np.set_printoptions(threshold=100000) # Полный вывод массива numpy (без сокращений)

                # Преобразование отправляемых данных
                rand_images_str = str(rand_images)
                epochs_per_person_str = str(self.epochs_per_person)
                client_ID_str = str(self.client_ID)

                print('Задание успешно сгенерировано')
                print('Выполняется отправка задания клиенту...')
                self.log.writelines('Задание успешно сгенерировано\n')
                self.log.writelines('Выполняется отправка задания клиенту...\n')

                # Отправить ID клиента
                await(websocket.send(client_ID_str))

                # Отправка весов
                await self.send_weights(websocket)

                # Отправка ссылок на изображения и количества эпох обучения
                await websocket.send(rand_images_str)
                await websocket.send(epochs_per_person_str)

                print('Данные успешно отправлены')
                print()
                self.log.writelines('Данные успешно отправлены\n')
                self.log.writelines('\n')
                self.log.writelines('\n')

                self.client_ID = self.client_ID + 1
                self.epochs_computing = self.epochs_computing + self.epochs_per_person # Изменяем счётчик эпох с прошедшим обучением
            else:
                await websocket.send('NEURAL_NETWORK_ALREADY_TRAINED')
        elif(message == 'SEND RESULT'): # Если клиент отправил выполненное "задание"
            print('Выполняется приём выполненного задания от клиента...')
            self.log.writelines('Выполняется приём выполненного задания от клиента...\n')

            # Принимаем новые значения весов от клиента
            delta_weights = await self.get_weights(websocket)

            print('Приём от клиента выполненного задания успешно завершён')
            self.log.writelines('Приём от клиента выполненного задания успешно завершён\n')

            self.change_global_weights(delta_weights) # Изменение глобальных значений весов
            self.total_received_tasks = self.total_received_tasks + 1 # Учёт количества клиентов, отправивших обновлённые значения весов

            message = 'Количество клиентов отправивших задания: ' + str(self.total_received_tasks)
            print(message)
            print()
            self.log.writelines(message)
            self.log.writelines('\n')
            self.log.writelines('\n')

            # print('total_received_tasks: ', self.total_received_tasks)
            # print('epochs_per_person: ', self.epochs_per_person)
            # print('count_of_epochs: ', self.epochs_total)
            # print('self.total_received_tasks * self.epochs_per_person: ', self.total_received_tasks * self.epochs_per_person)

            if(self.total_received_tasks * self.epochs_per_person >= self.epochs_total): # Если все задания уже выполнены
                print('Распределённое обучение нейронной сети завершено')
                self.log.writelines('Распределённое обучение нейронной сети завершено\n')

                cm = self.test_nn() # Тестирование нейронной сети (проверка нейронной сети на тестовых данных)

                # Сохранение итоговой confusion matrix в файл
                file = open(self.confusion_matrix_filename, 'w')
                file.writelines(str(cm))
                file.writelines('\n')
                file.close()

                self.draw_metrix_data() # Вывод динамики изменения метрик во время обучения в виде графиков

                self.model.save_weights(self.weights_result_filename_h5)  # Сохранение окончательных значений весов в файл

                message1 = 'Окончательные значения весов сохранены в файле ' + self.weights_result_filename_h5
                print(message1)
                self.log.writelines(message1)
                self.log.writelines('\n')

                self.log.close() # Закрытие log файла
                exit(0) # Выход из программы
            else:
                message1 = 'Метрики нейронной сети после ' + str(self.total_received_tasks) + '-ого клиента'
                message2 = 'Количество пройденных эпох обучения: ' + str(self.total_received_tasks * self.epochs_per_person)
                print(message2)
                print(message1)
                print()
                self.log.writelines(message2)
                self.log.writelines('\n')
                self.log.writelines(message1)
                self.log.writelines('\n')
                self.log.writelines('\n')

                self.test_nn() # Тестирование нейронной сети (проверка нейронной сети на тестовых данных)

    # Метод для вывода графиков обучения нейронной сети
    def draw_metrix_data(self):
        epochs = [(i + 1) * self.epochs_per_person for i in range(math.floor(self.epochs_total / self.epochs_per_person))]

        plt.figure(figsize=(12, 9))

        plt.subplot(2, 3, 1)
        # plt.title('Значение метрики precision (macro) на каждой эпохе')
        plt.xlabel('Эпохи')
        plt.ylabel('precision (macro)')
        plt.grid()
        plt.plot(epochs, self.precision_macro)

        plt.subplot(2, 3, 2)
        # plt.title('Значение метрики precision (micro) на каждой эпохе')
        plt.xlabel('Эпохи')
        plt.ylabel('precision (mmicro)')
        plt.grid()
        plt.plot(epochs, self.precision_micro)

        plt.subplot(2, 3, 3)
        # plt.title('Значение метрики precision (weighted) на каждой эпохе')
        plt.xlabel('Эпохи')
        plt.ylabel('precision (weighted)')
        plt.grid()
        plt.plot(epochs, self.precision_weighted)

        plt.subplot(2, 3, 4)
        # plt.title('Значение метрики recall (macro) на каждой эпохе')
        plt.xlabel('Эпохи')
        plt.ylabel('recall (macro)')
        plt.grid()
        plt.plot(epochs, self.recall_macro)

        plt.subplot(2, 3, 5)
        # plt.title('Значение метрики recall (micro) на каждой эпохе')
        plt.xlabel('Эпохи')
        plt.ylabel('recall (micro)')
        plt.grid()
        plt.plot(epochs, self.recall_micro)

        plt.subplot(2, 3, 6)
        # plt.title('Значение метрики recall (weighted) на каждой эпохе')
        plt.xlabel('Эпохи')
        plt.ylabel('recall (weighted)')
        plt.grid()
        plt.plot(epochs, self.recall_weighted)

        plt.savefig('precision_recall_metrics.png')
        plt.show()

        plt.xlabel('Эпохи')
        plt.ylabel('accuracy')
        plt.grid()
        plt.plot(epochs, self.accuracy)

        plt.savefig('accuracy_metric.png')
        plt.show()

# Создание экземпляра класса Server и задание основных входных данных
my_server = Server()
# my_server.change_data()

# Если файл с весами существует, то загрузить его, иначе пройти тестовое обучение
check_file = os.path.exists(my_server.weights_filename_h5)
if(check_file):
    my_server.load_base_weights()
else:
    my_server.get_base_weights()

# Запуск сервера
try:
    main_server = websockets.serve(my_server.data_exchange, my_server.ip_address, my_server.port)
    asyncio.get_event_loop().run_until_complete(main_server)
    asyncio.get_event_loop().run_forever()
except Exception:
    my_server.log.close()