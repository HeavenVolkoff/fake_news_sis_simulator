from __future__ import annotations

# Internal
import typing as T
from enum import IntEnum, auto, unique


@unique
class Topology(IntEnum):
    Star = auto()
    Clique = auto()


class Simulador:
    def __init__(
        self, users: T.Sequence[T.Tuple[bool]], *, rnd: bool, topology: Topology = Topology.Clique
    ):
        self.users = users
        self.event_queue = []


__all__ = ("Simulador",)
