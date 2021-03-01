from __future__ import annotations

# Internal
import typing as T
from enum import IntEnum, auto, unique
from heapq import heappop, heappush
from logging import getLogger
from operator import add
from functools import reduce

# External
from numpy.random import randint, exponential

logger = getLogger(name=__name__)


@unique
class Topology(IntEnum):
    Star = auto()
    Clique = auto()


@unique
class EventType(IntEnum):
    fake = auto()
    truth = auto()


@unique
class EventOrigin(IntEnum):
    external = auto()
    internal = auto()


class Event(T.NamedTuple):
    # WARNING: Delta must be first element
    delta: float
    type: EventType
    origin: EventOrigin
    user_id: int


class Simulator:
    def __init__(
        self,
        users: T.Sequence[T.MutableSequence[bool]],
        *,
        rnd: bool,
        topology: Topology = Topology.Clique,
        fake_rate_heuristic: T.Callable[[int], T.Union[int, float]],
        truth_rate_heuristic: T.Callable[[int], T.Union[int, float]],
        internal_fake_transmission_rate: float,
        external_fake_transmission_rate: float,
        internal_truth_transmission_rate: float,
        external_truth_transmission_rate: float,
    ):
        self.rnd = rnd
        self.clock: float = 0
        self.topology = topology
        self.users_timeline = users
        self.fake_rate_heuristic = fake_rate_heuristic
        self.truth_rate_heuristic = truth_rate_heuristic
        self.external_fake_transmission_rate = external_fake_transmission_rate
        self.internal_fake_transmission_rate = internal_fake_transmission_rate
        self.internal_truth_transmission_rate = internal_truth_transmission_rate
        self.external_truth_transmission_rate = external_truth_transmission_rate

    def run(self, duration: float) -> None:
        end = self.clock + duration
        while self.clock <= end:
            event_queue: T.List[Event] = []
            self.gen_events(event_queue)
            event = heappop(event_queue)

            timeline = self.users_timeline[event.user_id]
            if self.rnd:
                timeline[randint(len(timeline))] = event.type == EventType.truth
            else:
                timeline.append(event.type == EventType.truth)
                timeline.pop()

    def gen_events(self, event_queue: T.List[Event]):
        # External events
        for (timeline_id,) in enumerate(self.users_timeline):
            heappush(
                event_queue,
                Event(
                    exponential(self.external_fake_transmission_rate),
                    EventType.fake,
                    EventOrigin.external,
                    timeline_id,
                ),
            )
            heappush(
                event_queue,
                Event(
                    exponential(self.external_truth_transmission_rate),
                    EventType.truth,
                    EventOrigin.external,
                    timeline_id,
                ),
            )

        # Internal events
        for current in self.users_timeline:
            fake_count = reduce(add, current)
            truth_count = len(current) - fake_count
            for timeline_id, user in enumerate(self.users_timeline):
                if user == current:
                    continue

                if fake_count > 0:
                    heappush(
                        event_queue,
                        Event(
                            exponential(self.fake_rate_heuristic(fake_count)),
                            EventType.fake,
                            EventOrigin.internal,
                            timeline_id,
                        ),
                    )

                if truth_count > 0:
                    heappush(
                        event_queue,
                        Event(
                            exponential(self.truth_rate_heuristic(truth_count)),
                            EventType.truth,
                            EventOrigin.internal,
                            timeline_id,
                        ),
                    )


__all__ = ("Simulator",)
