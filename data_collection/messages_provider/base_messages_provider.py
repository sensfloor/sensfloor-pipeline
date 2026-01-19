import threading
from abc import ABC, abstractmethod
from queue import Queue
from typing import Any, Protocol, Self


class MessagesProvider(Protocol):
    messages_queue: Queue[dict[str, Any]]

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_messages(self) -> list[dict[str, Any]]: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, value, traceback) -> bool: ...  # noqa: ANN001


class BaseMessagesProvider(ABC, MessagesProvider):
    def __init__(self) -> None:
        self.messages_queue = Queue()
        self._stop_event = threading.Event()
        self._thread = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, value, traceback) -> bool:  # noqa: ANN001
        self.stop()
        return False

    @abstractmethod
    def _run(self) -> None:
        pass

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            error_message = "Can't stop not started thread"
            raise RuntimeError(error_message)

        self._stop_event.set()
        self._thread.join()

    def get_messages(self) -> list[dict[str, Any]]:
        messages = []
        while not self.messages_queue.empty():
            messages.append(self.messages_queue.get())
        return messages
