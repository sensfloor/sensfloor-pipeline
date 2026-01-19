import asyncio
import contextlib
import json
import threading
from queue import Full, Queue
from typing import Self

from websockets.asyncio.server import serve
from websockets.server import WebSocketServerProtocol

_STOP = object()


class Websocket:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self._host = host
        self._port = port

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._queue = Queue(maxsize=1)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, value, traceback) -> bool:  # noqa: ANN001
        self.stop()
        return False

    def start(self) -> None:
        if self._thread is not None:
            error_message = "Websocket already running"
            raise RuntimeError(error_message)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def send_poses(self, poses: dict) -> None:
        try:
            self._queue.put_nowait(poses)
        except Full:
            _ = self._queue.get_nowait()
            self._queue.put_nowait(poses)

    def stop(self) -> None:
        if self._thread is None:
            error_message = "Can't stop not started thread"
            raise RuntimeError(error_message)

        self._stop_event.set()

        with contextlib.suppress(Full):
            self._queue.put_nowait(_STOP)

        self._thread.join()
        self._thread = None

    def _run(self) -> None:
        asyncio.run(self._async_main())

    async def _async_main(self) -> None:
        async with serve(self._handler, self._host, self._port):
            await asyncio.to_thread(self._stop_event.wait)

    async def _handler(self, websocket: WebSocketServerProtocol) -> None:
        while not self._stop_event.is_set():
            item = await asyncio.to_thread(self._queue.get)

            if item is _STOP:
                print("Stop")
                break

            await websocket.send(json.dumps(item))
