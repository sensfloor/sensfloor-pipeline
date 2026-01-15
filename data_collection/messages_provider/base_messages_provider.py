import threading
from abc import ABC, abstractmethod
from queue import Queue
from typing import Any, Protocol


class MessagesProvider(Protocol):
    messages_queue: Queue[dict[str, Any]]

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_messages(self) -> list[dict[str, Any]]: ...


class BaseMessagesProvider(ABC, MessagesProvider):
    def __init__(self) -> None:
        self.messages_queue = Queue()
        self._stop_event = threading.Event()
        self._thread = None

    @abstractmethod
    def _run(self) -> None:
        pass

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run)
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
