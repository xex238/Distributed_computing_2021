import asyncio
import websockets
import random

from keras.datasets import cifar10

main_port = 8771  # Порт для подключения клиентов

class Server:
    def __init__(self):
        train_images = 50000  # Количество изображений для обучения
        count_of_epochs = 10000  # Общее количество эпох обучения
        epochs_per_person = 10  # Количество эпох обучения для одного узла
        part_of_images = 0.01  # Доля изображений для хоста для обучение нейронной сети (изображения генерируются
        # случайным образом)

        weights = None  # Глобальные значения весов модели

        # Параметры для обучения
        loss = 'categorical_crossentropy'
        optimizer = 'adam'
        metrics = ['accuracy']

    # Метод для выдачи задания клиенту
    def get_task(self):
        # Получить случайным образом изображения для обучения для хоста
        rand_images = []
        l = list(range(1, 50001))
        random.shuffle(l)
        for i in range(round(self.part_of_images * self.train_images)):
            rand_images.append(l[i])

        message = 'weights:\n' + self.weights + '\n'
        message += 'rand_images:\n' + rand_images + '\n'
        message += 'count_of_epochs:\n' + self.epochs_per_person

    def change_global_weights(self, delta_weights):
        for i1 in range(len(self.weights)):
            for i2 in range(len(self.weights[0])):
                for i3 in range(len(self.weights[0][0])):
                    for i4 in range(len(self.weights[0][0][0])):
                        for i5 in range(len(self.weights[0][0][0][0])):
                            self.weights[i1][i2][i3][i4][i5] += delta_weights[i1][i2][i3][i4][i5]


    async def data_exchange(self, websocket):
        message = await websocket.recv()
        print(message)
        message_split = message.split('\n')

        if(message_split[0] == 'GET TASK'):
            await websocket.send(self.get_task())
        if(message_split[0] == 'SEND RESULT'):
            self.change_global_weights(message_split[1])

my_server = Server()

main_server = websockets.serve(my_server.data_exchange, "localhost", main_port)
asyncio.get_event_loop().run_until_complete(main_server)
asyncio.get_event_loop().run_forever()