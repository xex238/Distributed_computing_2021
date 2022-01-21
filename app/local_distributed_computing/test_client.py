import asyncio
import websockets

ip_address = 'localhost'
main_port = 8772

async def send_data():
    uri = "ws://" + ip_address + ":" + str(main_port)
    async with websockets.connect(uri) as websocket:
        await websocket.send('Hello world!')

answer = 'n'
# answer = input('Изменить значения по умолчанию? (y|n) ')
if (answer == 'y'):
    message = 'Введите ip-адрес сервера: '
    ip_address = (input(message))
    message = 'Введите номер порта сервера: '
    main_port = int(input(message))

asyncio.get_event_loop().run_until_complete(send_data())