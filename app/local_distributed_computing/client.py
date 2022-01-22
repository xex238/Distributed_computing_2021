import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10

import asyncio
import websockets
import json
import ast
import math
import time

class Client:
    def __init__(self):
        self.ip_address = 'localhost' # ip адрес сервера
        self.port = 8771 # Порт сервера для подключения
        self.ID = 1 # ID клиента (выдаётся сервером)

        self.GET_TASK_MESSAGE = 'GET TASK' # Сообщение для запроса "задания"
        self.SEND_RESULT_MESSAGE = 'SEND RESULT' # Сообщение для отправки результата после обучения нейронной сети
        self.NEURAL_NETWORK_ALREADY_TRAINED = 'NEURAL_NETWORK_ALREADY_TRAINED' # Нейронная сеть уже обучена

        self.loss = 'categorical_crossentropy'
        self.optimizer = 'adam'
        self.metrics = ['accuracy']

        self.start_weights = None # Список весов до начала обучения нейронной сети
        self.end_weights = None # Список весов после начала обучения нейронной сети
        self.delta_weights = None # Разница между весами после и до начала обучения нейронной сети
        self.total_epochs = None # Количество эпох обучения
        self.learning_images = None # Список номеров изображений для обучения

        self.model = None # Модель для обучения нейронной сети

        # Данные датасета
        self.X_train = []
        self.y_train = []

        self.class_num = None

        self.log_name = 'client_log.txt' # Имя файла для логов
        self.log = None # Файл с логами

    # Изменить некоторые начальные значения
    def change_data(self):
        answer = input('Изменить значения по умолчанию? (y|n) ')
        if(answer == 'y'):
            message = 'Введите ip-адрес, на котором будем открывать websocket соединение: '
            self.ip_address = (input(message))
            message = 'Введите номер порта, на котором будем открывать websocket соединение: '
            self.port = int(input(message))

    # Загрузка датасета cifar-10 и подготовка данных
    def data_preparing(self):
        # loading in the data
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        # normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        self.class_num = y_test.shape[1]

        return X_train, y_train, X_test, y_test

    # Метод для создания модели нейронной сети
    def create_model(self):
        print('Идёт создание модели нейронной сети...')
        self.log.writelines('Идёт создание модели нейронной сети...')

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

        print('Модель нейронной сети успешно создана')
        print()
        self.log.writelines('Модель нейронной сети успешно создана')
        self.log.writelines('\n')

    # Метод для запуска обучения нейронной сети
    def learning(self):
        self.create_model()

        print('Загрузка полученных от сервера весов в модель нейронной сети...')
        self.log.writelines('Загрузка полученных от сервера весов в модель нейронной сети...')
        self.model.set_weights(self.start_weights) # Могут быть проблемы при преобразовании весов!!!
        print('Веса успешно загружены в модель нейронной сети')
        print()
        self.log.writelines('Веса успешно загружены в модель нейронной сети')
        self.log.writelines('\n')

        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        print('model.summary:')
        print(self.model.summary())

        # Обучение
        print('Идёт обучение нейронной сети...')
        self.log.writelines('Идёт обучение нейронной сети...')
        history = self.model.fit(self.X_train, self.y_train, epochs=self.total_epochs, verbose=1)
        print('Обучение нейронной сети успешно завершено')
        print()
        self.log.writelines('Обучение нейронной сети успешно завершено')
        self.log.writelines('\n')

        self.end_weights = self.model.get_weights()
        self.compute_delta_weights()

    # Расчитать значения изменения весов нейронной сети
    def compute_delta_weights(self):
        print('Выполняется расчёт градиентов (изменений весов нейронной сети)')
        self.log.writelines('Выполняется расчёт градиентов (изменений весов нейронной сети)')

        for i2 in range(len(self.start_weights[0])):
            for i3 in range(len(self.start_weights[0][i2])):
                for i4 in range(len(self.start_weights[0][i2][i3])):
                    for i5 in range(len(self.start_weights[0][i2][i3][i4])):
                        # print(0, ', ', i2, ', ', i3, ', ', i4, ', ', i5)
                        # print(self.weights[0][i2][i3][i4][i5])
                        # print()
                        self.start_weights[0][i2][i3][i4][i5] = self.end_weights[0][i2][i3][i4][i5] - self.start_weights[0][i2][i3][i4][i5]

        for i1 in range(5):
            for i2 in range(len(self.start_weights[i1 + 1])):
                # print(i1 + 1, ', ', i2)
                # print(self.weights[i1 + 1][i2])
                # print()
                # await websocket.send(str(self.weights[i1 + 1][i2]))
                self.start_weights[i1 + 1][i2] = self.end_weights[i1 + 1][i2] - self.start_weights[i1 + 1][i2]

        for i2 in range(len(self.start_weights[6])):
            for i3 in range(len(self.start_weights[6][i2])):
                for i4 in range(len(self.start_weights[6][i2][i3])):
                    for i5 in range(len(self.start_weights[6][i2][i3][i4])):
                        # print(6, ', ', i2, ', ', i3, ', ', i4, ', ', i5)
                        # print(self.weights[6][i2][i3][i4][i5])
                        # print()
                        # await websocket.send(str(self.weights[6][i2][i3][i4][i5]))
                        self.start_weights[6][i2][i3][i4][i5] = self.end_weights[6][i2][i3][i4][i5] - self.start_weights[6][i2][i3][i4][i5]

        for i1 in range(5):
            for i2 in range(len(self.start_weights[i1 + 7])):
                # print(i1 + 7, ', ', i2)
                # print(self.weights[i1 + 7][i2])
                # print()
                # await websocket.send(str(self.weights[i1 + 7][i2]))
                self.start_weights[i1 + 7][i2] = self.end_weights[i1 + 7][i2] - self.start_weights[i1 + 7][i2]

        for i2 in range(len(self.start_weights[12])):
            for i3 in range(len(self.start_weights[12][i2])):
                # print(12, ', ', i2, ', ', i3)
                # print(self.weights[12][i2][i3])
                # print()
                # await websocket.send(str(self.weights[12][i2][i3]))
                self.start_weights[12][i2][i3] = self.end_weights[12][i2][i3] - self.start_weights[12][i2][i3]

        for i1 in range(5):
            for i2 in range(len(self.start_weights[i1 + 13])):
                # print(i1 + 13, ', ', i2)
                # print(self.weights[i1 + 13][i2])
                # print()
                # await websocket.send(str(self.weights[i1 + 13][i2]))
                self.start_weights[i1 + 13][i2] = self.end_weights[i1 + 13][i2] - self.start_weights[i1 + 13][i2]

        for i2 in range(len(self.start_weights[18])):
            for i3 in range(len(self.start_weights[18][i2])):
                # print(18, ', ', i2, ', ', i3)
                # print(self.weights[18][i2][i3])
                # print()
                # await websocket.send(str(self.weights[18][i2][i3]))
                self.start_weights[18][i2][i3] = self.end_weights[18][i2][i3] - self.start_weights[18][i2][i3]

        for i1 in range(5):
            for i2 in range(len(self.start_weights[i1 + 19])):
                # print(i1 + 19, ', ', i2)
                # print(self.weights[i1 + 19][i2])
                # print()
                # await websocket.send(str(self.weights[i1 + 19][i2]))
                self.start_weights[i1 + 19][i2] = self.end_weights[i1 + 19][i2] - self.start_weights[i1 + 19][i2]

        for i2 in range(len(self.start_weights[24])):
            for i3 in range(len(self.start_weights[24][i2])):
                # print(24, ', ', i2, ', ', i3)
                # print(self.weights[24][i2][i3])
                # print()
                # await websocket.send(str(self.weights[24][i2][i3]))
                self.start_weights[24][i2][i3] = self.end_weights[24][i2][i3] - self.start_weights[24][i2][i3]

        for i2 in range(len(self.start_weights[25])):
            # print(25, ', ', i2)
            # print(self.weights[25][i2])
            # print()
            # await websocket.send(str(self.weights[25][i2]))
            self.start_weights[25][i2] = self.end_weights[25][i2] - self.start_weights[25][i2]

        self.delta_weights = self.start_weights

        print('Расчёт градиентов (изменений весов нейронной сети) успешно выполнен')
        print()
        self.log.writelines('Расчёт градиентов (изменений весов нейронной сети) успешно выполнен')
        self.log.writelines('\n')

    # Метод для отправки весов
    async def send_weights(self, websocket):
        start_time = time.time()
        print('Идёт отправка весов. Завершено на 0%')
        self.log.writelines('Идёт отправка весов. Завершено на 0%')
        total_weights = 4250058
        send_weights = 0
        send_persentage = 0

        for i2 in range(len(self.delta_weights[0])):
            for i3 in range(len(self.delta_weights[0][i2])):
                for i4 in range(len(self.delta_weights[0][i2][i3])):
                    for i5 in range(len(self.delta_weights[0][i2][i3][i4])):
                        await websocket.send(str(self.delta_weights[0][i2][i3][i4][i5]))
                        send_weights = send_weights + 1

        for i1 in range(5):
            for i2 in range(len(self.delta_weights[i1 + 1])):
                await websocket.send(str(self.delta_weights[i1 + 1][i2]))
                send_weights = send_weights + 1

        for i2 in range(len(self.delta_weights[6])):
            for i3 in range(len(self.delta_weights[6][i2])):
                for i4 in range(len(self.delta_weights[6][i2][i3])):
                    for i5 in range(len(self.delta_weights[6][i2][i3][i4])):
                        await websocket.send(str(self.delta_weights[6][i2][i3][i4][i5]))
                        send_weights = send_weights + 1

        for i1 in range(5):
            for i2 in range(len(self.delta_weights[i1 + 7])):
                await websocket.send(str(self.delta_weights[i1 + 7][i2]))
                send_weights = send_weights + 1

        for i2 in range(len(self.delta_weights[12])):
            for i3 in range(len(self.delta_weights[12][i2])):
                await websocket.send(str(self.delta_weights[12][i2][i3]))
                send_weights = send_weights + 1
                if(send_persentage < math.floor(send_weights / total_weights * 100)):
                    send_persentage = send_persentage + 1
                    message = 'Идёт отправка весов на сервер. Завершено на ' + str(send_persentage) + '%'
                    print(message)
                    self.log.writelines(message)

        for i1 in range(5):
            for i2 in range(len(self.delta_weights[i1 + 13])):
                await websocket.send(str(self.delta_weights[i1 + 13][i2]))
                send_weights = send_weights + 1

        for i2 in range(len(self.delta_weights[18])):
            for i3 in range(len(self.delta_weights[18][i2])):
                await websocket.send(str(self.delta_weights[18][i2][i3]))
                send_weights = send_weights + 1

        for i1 in range(5):
            for i2 in range(len(self.delta_weights[i1 + 19])):
                await websocket.send(str(self.delta_weights[i1 + 19][i2]))
                send_weights = send_weights + 1

        for i2 in range(len(self.delta_weights[24])):
            for i3 in range(len(self.delta_weights[24][i2])):
                await websocket.send(str(self.delta_weights[24][i2][i3]))
                send_weights = send_weights + 1

        for i2 in range(len(self.delta_weights[25])):
            await websocket.send(str(self.delta_weights[25][i2]))
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
        print('Загрузка весов с сервера. Завершено на 0%')
        self.log.writelines('Загрузка весов с сервера. Завершено на 0%')
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
                        # print(0, ', ', i2, ', ', i3, ', ', i4, ', ', i5)
                        weights3.append(float(await websocket.recv()))
                        get_weights = get_weights + 1
                    weights2.append(np.array(weights3))
                weights1.append(np.array(weights2))
            weights0.append(np.array(weights1))
        weights.append(np.array(weights0))

        for i1 in range(5):
            weights1 = []
            for i2 in range(32):
                # print(i1 + 1, ', ', i2)
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
                        # print(6, ', ', i2, ', ', i3, ', ', i4, ', ', i5)
                        weights3.append(float(await websocket.recv()))
                        get_weights = get_weights + 1
                    weights2.append(np.array(weights3))
                weights1.append(np.array(weights2))
            weights0.append(np.array(weights1))
        weights.append(np.array(weights0))

        for i1 in range(5):
            weights1 = []
            for i2 in range(64):
                # print(i1 + 7, ', ', i2)
                weights1.append(float(await websocket.recv()))
                get_weights = get_weights + 1
            weights.append(np.array(weights1))

        weights0 = []
        for i2 in range(16384):
            weights1 = []
            for i3 in range(256):
                # print(12, ', ', i2, ', ', i3)
                weights1.append(float(await websocket.recv()))
                get_weights = get_weights + 1
                if(get_persentage < math.floor(get_weights / total_weights * 100)):
                    get_persentage = get_persentage + 1
                    message = 'Идёт загрузка весов с сервера. Завершено на ' + str(get_persentage) + '%'
                    print(message)
                    self.log.writelines(message)
            weights0.append(np.array(weights1))
        weights.append(np.array(weights0))

        for i1 in range(5):
            weights1 = []
            for i2 in range(256):
                # print(i1 + 13, ', ', i2)
                weights1.append(float(await websocket.recv()))
                get_weights = get_weights + 1
            weights.append(np.array(weights1))

        weights0 = []
        for i2 in range(256):
            weights1 = []
            for i3 in range(128):
                # print(18, ', ', i2, ', ', i3)
                weights1.append(float(await websocket.recv()))
                get_weights = get_weights + 1
            weights0.append(np.array(weights1))
        weights.append(np.array(weights0))

        for i1 in range(5):
            weights1 = []
            for i2 in range(128):
                # print(i1 + 19, ', ', i2)
                weights1.append(float(await websocket.recv()))
                get_weights = get_weights + 1
            weights.append(np.array(weights1))

        weights0 = []
        for i2 in range(128):
            weights1 = []
            for i3 in range(10):
                # print(24, ', ', i2, ', ', i3)
                weights1.append(float(await websocket.recv()))
                get_weights = get_weights + 1
            weights0.append(np.array(weights1))
        weights.append(np.array(weights0))

        weights0 = []
        for i2 in range(10):
            # print(25, ', ', i2)
            weights0.append(float(await websocket.recv()))
            get_weights = get_weights + 1
        weights.append(np.array(weights0))

        self.start_weights = weights

        end_time = time.time()
        message = 'Время загрузки весов составило ' + str(round(end_time - start_time, 2)) + ' минут'

        print('Веса успешно загружены')
        print(message)
        print()
        self.log.writelines('Веса успешно загружены')
        self.log.writelines(message)
        self.log.writelines('\n')

    # Метод для вывода метаданных о весах
    def print_weights_metadata(self):
        print('type(weights): ', type(self.start_weights))

        print('type(weights[0]): ', type(self.start_weights[0]))
        print('type(weights[0][0]): ', type(self.start_weights[0][0]))
        print('type(weights[0][0][0]): ', type(self.start_weights[0][0][0]))
        print('type(weights[0][0][0][0]): ', type(self.start_weights[0][0][0][0]))
        print('type(weights[0][0][0][0][0]): ', type(self.start_weights[0][0][0][0][0]))

        print('type(weights[1]): ', type(self.start_weights[1]))
        print('type(weights[1][0]): ', type(self.start_weights[1][0]))

        print('type(weights[6]): ', type(self.start_weights[6]))
        print('type(weights[6][0]): ', type(self.start_weights[6][0]))
        print('type(weights[6][0][0]): ', type(self.start_weights[6][0][0]))
        print('type(weights[6][0][0][0]): ', type(self.start_weights[6][0][0][0]))
        print('type(weights[6][0][0][0][0]): ', type(self.start_weights[6][0][0][0][0]))

        print('type(weights[7]): ', type(self.start_weights[7]))
        print('type(weights[7][0]): ', type(self.start_weights[7][0]))

        print('type(weights[12]): ', type(self.start_weights[12]))
        print('type(weights[12][0]): ', type(self.start_weights[12][0]))
        print('type(weights[12][0][0]): ', type(self.start_weights[12][0][0]))

        print('type(weights[13]): ', type(self.start_weights[13]))
        print('type(weights[13][0]): ', type(self.start_weights[13][0]))

        print('type(weights[18]): ', type(self.start_weights[18]))
        print('type(weights[18][0]): ', type(self.start_weights[18][0]))
        print('type(weights[18][0][0]): ', type(self.start_weights[18][0][0]))

        print('type(weights[19]): ', type(self.start_weights[19]))
        print('type(weights[19][0]): ', type(self.start_weights[19][0]))

        print('type(weights[24]): ', type(self.start_weights[24]))
        print('type(weights[24][0]): ', type(self.start_weights[24][0]))
        print('type(weights[24][0][0]): ', type(self.start_weights[24][0][0]))

        print('type(weights[25]): ', type(self.start_weights[25]))
        print('type(weights[25][0]): ', type(self.start_weights[25][0]))

    # Метод для запроса "задания" у сервера, обработки и запуска данного задания
    async def data_request(self):
        print('Обращение к сервверу за данными')

        uri = "ws://" + self.ip_address + ":" + str(self.port)
        async with websockets.connect(uri) as websocket:
            await websocket.send(self.GET_TASK_MESSAGE) # Отправляем запрос о "задании" на сервер

            message = await websocket.recv()

            if(message != 'NEURAL_NETWORK_ALREADY_TRAINED'):
                # Приём значения ID клиента
                self.ID = int(message)
                self.log_name = 'client_log' + message + '.txt'
                self.log = open(self.log_name, 'w')

                await self.get_weights(websocket)

                images_str = await websocket.recv() # Получение списка с изображениями для обучения
                total_epochs_str = await websocket.recv() # Получение значения количества эпох для обучения

                print('Данные от сервера успешно получены')
                self.log.writelines('Данные от сервера успешно получены')

                np.set_printoptions(threshold=100000)

                # Конвертация полученных данных из формата str в формат list
                images = json.loads(images_str)
                self.total_epochs = int(total_epochs_str)

                X_train, y_train, X_test, y_test = self.data_preparing()
                for i in range(len(images)):
                    self.X_train.append(X_train[i])
                    self.y_train.append(y_train[i])

                self.X_train = np.array(self.X_train)
                self.y_train = np.array(self.y_train)

                # self.print_weights_metadata()
                # print(self.start_weights)

                print('Данные успешно конвертированы')
                print()
                self.log.writelines('Данные успешно конвертированы')
                self.log.writelines('\n')

    # Метод для отправки результата (изменение весов при обучении, delta)
    async def send_result(self):
        print('Отправка результатов (градиентов) на сервер')
        self.log.writelines('Отправка результатов (градиентов) на сервер')

        uri = "ws://" + self.ip_address + ":" + str(self.port)
        async with websockets.connect(uri) as websocket:
            await websocket.send(self.SEND_RESULT_MESSAGE) # Отправляем сообщение об отправлке результатов на сервер
            await self.send_weights(websocket)

        print('Данные успешно отправлены на сервер')
        self.log.writelines('Данные успешно отправлены на сервер')
        self.log.close()

my_client = Client()
# my_client.change_data()

asyncio.get_event_loop().run_until_complete(my_client.data_request())
# my_client.print_weights_metadata()
my_client.learning()
asyncio.get_event_loop().run_until_complete(my_client.send_result())