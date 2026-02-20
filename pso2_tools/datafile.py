from collections.abc import Iterable
from typing import Protocol


class DataFile(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def data(self) -> bytes: ...


class DataFileSource(Protocol):
    def get_files(self) -> Iterable[DataFile]: ...

    def glob(self, pattern: str) -> Iterable[DataFile]: ...
