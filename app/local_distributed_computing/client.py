import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10

import asyncio
import websockets

class Client:
    def __init__(self):
        main_port = 8771 # Порт сервера для подключения
        GET_TASK_MESSAGE = 'GET TASK' # Сообщение для запроса "задания"
        SEND_RESULT_MESSAGE = 'SEND RESULT' # Сообщение для отправки результата после обучения нейронной сети

        loss = 'categorical_crossentropy'
        optimizer = 'adam'
        metrics = ['accuracy']

        start_weights = None # Список весов до начала обучения нейронной сети
        end_weights = None # Список весов после начала обучения нейронной сети
        total_epochs = None # Количество эпох обучения
        learning_images = None # Список номеров изображений для обучения

        model = None # Модель для обучения нейронной сети

        # Данные датасета
        X_train = None
        X_test = None
        X_train = None
        X_test = None

        y_train = None
        y_test = None
        class_num = None

    # Загрузка датасета cifar-10 и подготовка данных
    def data_preparing(self):
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

    # Метод для создания модели нейронной сети
    def create_model(self):
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

    # Метод для обработки "задания" (полученных данных) с сервера
    def data_parser(self, message):
        print()

    # Метод для запуска обучения нейронной сети
    def learning(self):
        self.data_preparing()
        self.create_model()

        self.model.set_weights(self.start_weights) # Могут быть проблемы при преобразовании весов!!!
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        print(self.model.summary())

        # Добавить собственно обучение модели (нейронной сети)
        # Добавить сохранение весов обученной модели (нейронной сети)

    # Тестовый метод для проверки соединения по протоколу websocket
    async def hello(self):
        uri = "ws://localhost:" + str(self.main_port)
        async with websockets.connect(uri) as websocket:
            await websocket.send('Hello world!')

    # Метод для запроса "задания" у сервера, обработки и запуска данного задания
    async def data_request(self):
        uri = "ws://localhost:" + str(self.main_port)
        async with websockets.connect(uri) as websocket:
            await websocket.send(self.GET_TASK_MESSAGE) # Отправляем запрос о "задании" на сервер
            message = await websocket.recv() # Получение "задания" с сервера
            self.data_parser(message)

    # Метод для отправки результата (изменение весов при обучении, delta)
    def send_result(self):
        print()
        delta_weights = self.end_weights - self.start_weights
        message = self.SEND_RESULT_MESSAGE + "\n" + delta_weights

        uri = "ws://localhost:" + str(self.main_port)
        async with websockets.connect(uri) as websocket:
            await websocket.send(message)

my_client = Client()

asyncio.get_event_loop().run_until_complete(my_client.hello())
my_client.learning()