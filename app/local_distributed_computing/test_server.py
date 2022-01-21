import asyncio
import websockets

ip_address = 'localhost'
main_port = 8772

async def data_exchange(websocket):
    file = await websocket.recv()
    print('Сообщение 1: ', file)

answer = 'n'
# answer = input('Изменить значения по умолчанию? (y|n) ')
if (answer == 'y'):
    message = 'Введите ip-адрес, на котором будем открывать websocket соединение: '
    ip_address = (input(message))
    message = 'Введите номер порта, на котором будем открывать websocket соединение: '
    main_port = int(input(message))

main_server = websockets.serve(data_exchange, ip_address, main_port)
asyncio.get_event_loop().run_until_complete(main_server)
asyncio.get_event_loop().run_forever()