import asyncio
import websockets

main_port = 8771

async def hello():
    uri = "ws://localhost:" + str(main_port)
    async with websockets.connect(uri) as websocket:
        await websocket.send('Hello world!')

asyncio.get_event_loop().run_until_complete(hello())