import asyncio

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

URI = "ws://127.0.0.1:8765"


async def listen_forever() -> None:
    while True:
        try:
            async with connect(URI) as websocket:
                print("Connected.")
                async for message in websocket:
                    print(message)
        except (OSError, ConnectionClosed) as e:  # noqa: PERF203
            print(f"Disconnected ({e}). Reconnecting in 0.5s...")
            await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(listen_forever())
