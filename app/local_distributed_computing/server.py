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

from keras.datasets import cifar10

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
        self.count_of_epochs = 3  # Общее количество эпох обучения
        self.epochs_per_person = 1  # Количество эпох обучения для одного узла
        self.part_of_images = 0.01  # Доля изображений для хоста для обучение нейронной сети (изображения генерируются
        # случайным образом)

        self.weights_filename_h5 = 'weights.h5' # Файл для хранения начальных значений весов в формате h5
        self.weights_filename_txt = 'weights.txt' # Файл для хранения весов в txt формате
        self.weights_result_filename_h5 = 'weights_result.h5' # Файл для хранения весов после обучения в формате h5
        self.weights = None  # Глобальные значения весов модели

        self.model = None # Модель для тестирования нейронной сети

        # Данные датасета
        self.X_test = []
        self.y_test = []
        self.X_train = []
        self.y_train = []

        self.class_num = None

        # Параметры для обучения
        self.loss = 'categorical_crossentropy'
        self.optimizer = 'adam'
        self.metrics = ['accuracy']

        self.log_name = 'server_log.txt' # Имя файла для логов
        self.log = open(self.log_name, 'w') # Файл с логами

    # Изменить некоторые начальные значения
    def change_data(self):
        answer = input('Изменить значения по умолчанию? (y|n) ')
        if(answer == 'y'):
            message = 'Введите ip-адрес, на котором будем открывать websocket соединение: '
            self.ip_address = (input(message))
            message = 'Введите номер порта, на котором будем открывать websocket соединение: '
            self.port = int(input(message))
            message = 'Введите общее количество эпох обучения (по умолчанию ' + str(self.count_of_epochs) + ') '
            self.count_of_epochs = int(input(message))
            message = 'Введите количество эпох обучения для одного клиента (по умолчанию ' + str(self.epochs_per_person) + ') '
            self.epochs_per_person = int(input(message))
            message = 'Введите долю изображений из обучающей выборки, отправляемых клиенту (по умолчанию ' + str(self.part_of_images) + ') '
            self.part_of_images = float(input(message))

    # Начальное обучение нейронной сети на нескольких эпохах для получения начальных значений весов
    def get_base_weights(self):
        print('Создание модели и тестовое обучение на 1-ой эпохе')
        print()

        self.log.writelines('Создание модели и тестовое обучение на 1-ой эпохе')
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

        self.log.writelines('Модель создана и пройдено тестовое обучение')
        self.log.writelines(message1)
        self.log.writelines('Сервер запущен')
        self.log.writelines('\n')

    # Создание модели и загрузка начальных значений весов
    def load_base_weights(self):
        print('Создание модели и загрузка весов')
        print()

        self.log.writelines('Создание модели и загрузка весов')
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

        self.model.load_weights(self.weights_filename_h5)
        self.weights = self.model.get_weights()

        message1 = 'Начальные значения загружены из файла ' + self.weights_filename_h5

        print('Модель создана и веса загружены')
        print(message1)
        print('Сервер запущен')
        print()

        self.log.writelines('Модель создана и веса загружены')
        self.log.writelines(message1)
        self.log.writelines('Сервер запущен')
        self.log.writelines('\n')

    # Метод для изменения глобального значения весов
    def change_global_weights(self, delta_weights):
        print('Корректировка глобальных весов нейронной сети')

        for i2 in range(len(self.weights[0])):
            for i3 in range(len(self.weights[0][i2])):
                for i4 in range(len(self.weights[0][i2][i3])):
                    for i5 in range(len(self.weights[0][i2][i3][i4])):
                        # print(0, ', ', i2, ', ', i3, ', ', i4, ', ', i5)
                        # print(self.weights[0][i2][i3][i4][i5])
                        # print()
                        self.weights[0][i2][i3][i4][i5] = self.weights[0][i2][i3][i4][i5] + delta_weights[0][i2][i3][i4][i5]

        for i1 in range(5):
            for i2 in range(len(self.weights[i1 + 1])):
                # print(i1 + 1, ', ', i2)
                # print(self.weights[i1 + 1][i2])
                # print()
                # await websocket.send(str(self.weights[i1 + 1][i2]))
                self.weights[i1 + 1][i2] = self.weights[i1 + 1][i2] + delta_weights[i1 + 1][i2]

        for i2 in range(len(self.weights[6])):
            for i3 in range(len(self.weights[6][i2])):
                for i4 in range(len(self.weights[6][i2][i3])):
                    for i5 in range(len(self.weights[6][i2][i3][i4])):
                        # print(6, ', ', i2, ', ', i3, ', ', i4, ', ', i5)
                        # print(self.weights[6][i2][i3][i4][i5])
                        # print()
                        # await websocket.send(str(self.weights[6][i2][i3][i4][i5]))
                        self.weights[6][i2][i3][i4][i5] = self.weights[6][i2][i3][i4][i5] + delta_weights[6][i2][i3][i4][i5]

        for i1 in range(5):
            for i2 in range(len(self.weights[i1 + 7])):
                # print(i1 + 7, ', ', i2)
                # print(self.weights[i1 + 7][i2])
                # print()
                # await websocket.send(str(self.weights[i1 + 7][i2]))
                self.weights[i1 + 7][i2] = self.weights[i1 + 7][i2] + delta_weights[i1 + 7][i2]

        for i2 in range(len(self.weights[12])):
            for i3 in range(len(self.weights[12][i2])):
                # print(12, ', ', i2, ', ', i3)
                # print(self.weights[12][i2][i3])
                # print()
                # await websocket.send(str(self.weights[12][i2][i3]))
                self.weights[12][i2][i3] = self.weights[12][i2][i3] + delta_weights[12][i2][i3]

        for i1 in range(5):
            for i2 in range(len(self.weights[i1 + 13])):
                # print(i1 + 13, ', ', i2)
                # print(self.weights[i1 + 13][i2])
                # print()
                # await websocket.send(str(self.weights[i1 + 13][i2]))
                self.weights[i1 + 13][i2] = self.weights[i1 + 13][i2] + delta_weights[i1 + 13][i2]

        for i2 in range(len(self.weights[18])):
            for i3 in range(len(self.weights[18][i2])):
                # print(18, ', ', i2, ', ', i3)
                # print(self.weights[18][i2][i3])
                # print()
                # await websocket.send(str(self.weights[18][i2][i3]))
                self.weights[18][i2][i3] = self.weights[18][i2][i3] + delta_weights[18][i2][i3]

        for i1 in range(5):
            for i2 in range(len(self.weights[i1 + 19])):
                # print(i1 + 19, ', ', i2)
                # print(self.weights[i1 + 19][i2])
                # print()
                # await websocket.send(str(self.weights[i1 + 19][i2]))
                self.weights[i1 + 19][i2] = self.weights[i1 + 19][i2] + delta_weights[i1 + 19][i2]

        for i2 in range(len(self.weights[24])):
            for i3 in range(len(self.weights[24][i2])):
                # print(24, ', ', i2, ', ', i3)
                # print(self.weights[24][i2][i3])
                # print()
                # await websocket.send(str(self.weights[24][i2][i3]))
                self.weights[24][i2][i3] = self.weights[24][i2][i3] + delta_weights[24][i2][i3]

        for i2 in range(len(self.weights[25])):
            # print(25, ', ', i2)
            # print(self.weights[25][i2])
            # print()
            # await websocket.send(str(self.weights[25][i2]))
            self.weights[25][i2] = self.weights[25][i2] + delta_weights[25][i2]

        print('Глобальные значения весов успешно обновлены')

    # Тестирование обученной нейронной сети на тестовых данных
    def test_nn(self):
        print('Выполняется проверка нейронной сети на тестовых данных')
        self.log.writelines('Выполняется проверка нейронной сети на тестовых данных')

        self.model.set_weights(self.weights) # Загрузка глобальных значений весов в модель
        y_predict = self.model.predict(x=self.X_test) # Прогнозирование

        # Вывод полученных и действительных значений в консоль и в log файл
        message1 = 'y: ' + str(self.y_test)
        message2 = 'y_predict: ' + str(y_predict)
        print(message1)
        print(message2)
        print()
        self.log.writelines(message1)
        self.log.writelines(message2)
        self.log.writelines('\n')

        # Расчёт метрик
        '''
        try:
            accuracy = accuracy_score(self.y_test, y_predict)
            message = 'accuracy: ' + str(accuracy)
            print(message)
            self.log.writelines(message)
        except Exception:
            print('Возникла ошибка при подсчёте метрики accuracy_score')
            self.log.writelines('Возникла ошибка при подсчёте метрики accuracy_score')
        '''

        try:
            cvs = cross_val_score(y_predict, self.X_test, self.y_test)
            message = 'cross_val_score: ' + str(cvs)
            print(message)
            self.log.writelines(message)
        except Exception:
            print('Возникла ошибка при подсчёте метрики cross_val_score')
            self.log.writelines('Возникла ошибка при подсчёте метрики cross_val_score')

        try:
            precision = precision_score(self.y_test, y_predict, average=None)
            message = 'precision_score: ' + str(precision)
            print(message)
            self.log.writelines(message)
        except Exception:
            print('Возникла ошибка при подсчёте метрики precision_score')
            self.log.writelines('Возникла ошибка при подсчёте метрики precision_score')

        try:
            recall = recall_score(self.y_test, y_predict, average=None)
            message = 'recall_score: ' + str(recall)
            print(message)
            self.log.writelines(message)
        except Exception:
            print('Возникла ошибка при подсчёте метрики recall_score')
            self.log.writelines('Возникла ошибка при подсчёте метрики recall_score')

        try:
            cm = confusion_matrix(self.y_test, y_predict)
            message = 'confusion_matrix: ' + str(cm)
            print(message)
            self.log.writelines(message)
        except Exception:
            print('Возникла ошибка при подсчёте confusion_matrix')
            self.log.writelines('Возникла ошибка при подсчёте confusion_matrix')

        print('Проверка нейронной сети на тестовых данных завершена')
        print()
        self.log.writelines('Проверка нейронной сети на тестовых данных завершена')
        self.log.writelines('\n')

    # Метод для отправки весов
    async def send_weights(self, websocket):
        start_time = time.time()
        print('Идёт отправка весов. Завершено на 0%')
        self.log.writelines('Идёт отправка весов. Завершено на 0%')
        total_weights = 4250058
        send_weights = 0
        send_persentage = 0

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
        message = 'Время отправки весов составило ' + str(round(end_time - start_time, 2)) + ' минут'

        print('Веса успешно отправлены')
        print(message)
        print()
        self.log.writelines('Веса успешно отправлены')
        self.log.writelines(message)
        self.log.writelines('\n')

    # Метод для приёма весов
    async def get_weights(self, websocket):
        start_time = time.time()
        print('Загрузка весов от клиента... Завершено на 0%')
        self.log.writelines('Загрузка весов от клиента... Завершено на 0%')
        total_weights = 4250058
        get_weights = 0
        get_persentage = 0
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
        message = 'Время загрузки весов составило ' + str(round(end_time - start_time, 2)) + ' минут'

        print('Веса успешно загружены')
        print(message)
        print()
        self.log.writelines('Веса успешно загружены')
        self.log.writelines(message)
        self.log.writelines('\n')

        return weights

    # Метод для вывода метаданных о весах
    def print_weights_metadata(self):
        print('type(weights): ', type(self.weights))

        print('type(weights[0]): ', type(self.weights[0]))
        print('type(weights[0][0]): ', type(self.weights[0][0]))
        print('type(weights[0][0][0]): ', type(self.weights[0][0][0]))
        print('type(weights[0][0][0][0]): ', type(self.weights[0][0][0][0]))
        print('type(weights[0][0][0][0][0]): ', type(self.weights[0][0][0][0][0]))

        print('type(weights[1]): ', type(self.weights[1]))
        print('type(weights[1][0]): ', type(self.weights[1][0]))

        print('type(weights[6]): ', type(self.weights[6]))
        print('type(weights[6][0]): ', type(self.weights[6][0]))
        print('type(weights[6][0][0]): ', type(self.weights[6][0][0]))
        print('type(weights[6][0][0][0]): ', type(self.weights[6][0][0][0]))
        print('type(weights[6][0][0][0][0]): ', type(self.weights[6][0][0][0][0]))

        print('type(weights[7]): ', type(self.weights[7]))
        print('type(weights[7][0]): ', type(self.weights[7][0]))

        print('type(weights[12]): ', type(self.weights[12]))
        print('type(weights[12][0]): ', type(self.weights[12][0]))
        print('type(weights[12][0][0]): ', type(self.weights[12][0][0]))

        print('type(weights[13]): ', type(self.weights[13]))
        print('type(weights[13][0]): ', type(self.weights[13][0]))

        print('type(weights[18]): ', type(self.weights[18]))
        print('type(weights[18][0]): ', type(self.weights[18][0]))
        print('type(weights[18][0][0]): ', type(self.weights[18][0][0]))

        print('type(weights[19]): ', type(self.weights[19]))
        print('type(weights[19][0]): ', type(self.weights[19][0]))

        print('type(weights[24]): ', type(self.weights[24]))
        print('type(weights[24][0]): ', type(self.weights[24][0]))
        print('type(weights[24][0][0]): ', type(self.weights[24][0][0]))

        print('type(weights[25]): ', type(self.weights[25]))
        print('type(weights[25][0]): ', type(self.weights[25][0]))

    # websocket метод для прослушивания заданного порта и реакции на запросы
    async def data_exchange(self, websocket):
        self.print_weights_metadata() # Вывод на экран метаданных о переменной weights (типы данных списка)

        message = await websocket.recv() # Приём первоначального блока данных
        print(message)
        print()
        self.log.writelines(message)
        self.log.writelines('\n')

        if(message == 'GET TASK'): # Если клиент запросил "задание"
            if(self.count_of_epochs != 0): # Если не все "задания" ещё выданы
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

                # Отправить ID клиента
                await(websocket.send(client_ID_str))

                # Отправка весов
                await self.send_weights(websocket)

                # Отправка ссылок на изображения и количества эпох обучения
                await websocket.send(rand_images_str)
                await websocket.send(epochs_per_person_str)

                print('Данные успешно отправлены')
                print()
                self.log.writelines('Данные успешно отправлены')
                self.log.writelines('\n')

                self.client_ID = self.client_ID + 1
                self.count_of_epochs = self.count_of_epochs - self.epochs_per_person # Уменьшение общего количества эпох обучения после выдачи "задания"
            else:
                await websocket.send('NEURAL_NETWORK_ALREADY_TRAINED')
        elif(message == 'SEND RESULT'): # Если клиент отправил выполненное "задание"
            # Принимаем новые значения весов от клиента
            delta_weights = await self.get_weights(websocket)

            self.change_global_weights(delta_weights) # Изменение глобальных значений весов
            self.total_received_tasks = self.total_received_tasks + 1 # Учёт количества клиентов, отправивших обновлённые значения весов

            print('Веса от клиента успешно приняты')

            if(self.total_received_tasks * self.epochs_per_person == self.count_of_epochs): # Если все задания уже выполнены
                print('Распределённое обучение нейронной сети завершено')
                self.log.writelines('Распределённое обучение нейронной сети завершено')

                self.test_nn() # Тестирование нейронной сети (проверка нейронной сети на тестовых данных)

                self.model.save_weights(
                    self.weights_result_filename_h5)  # Сохранение окончательных значений весов в файл

                message1 = 'Окончательные значения весов сохранены в файле ' + self.weights_result_filename_h5
                print(message1)
                print()
                self.log.writelines(message1)
                self.log.writelines('\n')

                self.log.close() # Закрытие log файла
                exit(0) # Выход из программы
            else:
                message1 = 'Метрики нейронной сети после ' + str(self.total_received_tasks) + '-ого клиента'
                message2 = 'Количество пройденных эпох обучения: ' + str(self.total_received_tasks * self.epochs_per_person)
                print(message1)
                print(message2)
                print()
                self.log.writelines(message1)
                self.log.writelines(message2)
                self.log.writelines('\n')

                self.test_nn() # Тестирование нейронной сети (проверка нейронной сети на тестовых данных)

# Создание экземпляра класса Server и задание основных входных данных
my_server = Server()
# my_server.change_data()

# Если файл с весами существует, то загрузить его, иначе пройти тестовое обучение
check_file = os.path.exists(my_server.weights_filename_h5)
if(check_file):
    my_server.load_base_weights()
else:
    my_server.get_base_weights()
# my_server.print_weights_metadata()

# Запуск сервера
try:
    main_server = websockets.serve(my_server.data_exchange, my_server.ip_address, my_server.port)
    asyncio.get_event_loop().run_until_complete(main_server)
    asyncio.get_event_loop().run_forever()
except Exception:
    my_server.log.close()