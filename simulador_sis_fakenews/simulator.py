from __future__ import annotations

# Internal
import typing as T
from enum import IntEnum, auto, unique
from heapq import heappop, heappush
from operator import add
from functools import reduce
from collections import Counter

# External
from numpy.random import randint, exponential


@unique
class Topology(IntEnum):
    Star = auto()
    Clique = auto()


@unique
class EventType(IntEnum):
    Fake = 0
    Truth = 1


@unique
class EventOrigin(IntEnum):
    External = auto()
    Internal = auto()


class Event(T.NamedTuple):
    # WARNING: Delta must be first element
    delta: float
    type: EventType
    origin: EventOrigin
    user_id: int


class Simulator:
    def __init__(
        self,
        users: T.Sequence[T.MutableSequence[EventType]],
        *,
        rnd: bool = False,
        topology: Topology = Topology.Clique,
        fake_rate_heuristic: T.Callable[[int], T.Union[int, float]],
        truth_rate_heuristic: T.Callable[[int], T.Union[int, float]],
        internal_fake_transmission_rate: float,
        external_fake_transmission_rate: float,
        internal_truth_transmission_rate: float,
        external_truth_transmission_rate: float,
    ) -> None:
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

    def step(
        self,
    ) -> T.Generator[T.Tuple[float, Event, T.Counter[T.Tuple[EventType, ...]]], None, None]:
        while True:
            event_queue: T.List[Event] = []
            self.gen_events(event_queue)
            event = heappop(event_queue)

            # Update clock
            self.clock += event.delta

            timeline = self.users_timeline[event.user_id]
            if self.rnd:
                timeline[randint(len(timeline))] = event.type
            else:
                timeline.insert(0, event.type)
                timeline.pop()

            stats: T.Counter[T.Tuple[EventType, ...]] = Counter()
            for timeline in self.users_timeline:
                stats[tuple(timeline)] += 1

            yield self.clock, event, stats

    def debug(self, iterations: int) -> None:
        for pos, (time, event, stats) in enumerate(self.step()):
            if pos > iterations:
                break

            print(pos, event.type, event.origin, stats)

    def gen_events(self, event_queue: T.List[Event]) -> None:
        # External events
        for timeline_id, _ in enumerate(self.users_timeline):
            if self.external_fake_transmission_rate > 0:
                heappush(
                    event_queue,
                    Event(
                        exponential(1 / self.external_fake_transmission_rate),
                        EventType.Fake,
                        EventOrigin.External,
                        timeline_id,
                    ),
                )

            if self.external_truth_transmission_rate > 0:
                heappush(
                    event_queue,
                    Event(
                        exponential(1 / self.external_truth_transmission_rate),
                        EventType.Truth,
                        EventOrigin.External,
                        timeline_id,
                    ),
                )

        # Internal events
        for current_id, current in enumerate(self.users_timeline):
            fake_count = int(reduce(add, current))
            truth_count = len(current) - fake_count
            for timeline_id, user in enumerate(self.users_timeline):
                if timeline_id == current_id:
                    continue

                if fake_count > 0 and self.internal_fake_transmission_rate > 0:
                    heappush(
                        event_queue,
                        Event(
                            exponential(
                                1
                                / (
                                    self.fake_rate_heuristic(fake_count)
                                    * self.internal_fake_transmission_rate
                                )
                            ),
                            EventType.Fake,
                            EventOrigin.Internal,
                            timeline_id,
                        ),
                    )

                if truth_count > 0 and self.internal_truth_transmission_rate > 0:
                    heappush(
                        event_queue,
                        Event(
                            exponential(
                                1
                                / (
                                    self.truth_rate_heuristic(truth_count)
                                    * self.internal_truth_transmission_rate
                                )
                            ),
                            EventType.Truth,
                            EventOrigin.Internal,
                            timeline_id,
                        ),
                    )


__all__ = ("Simulator", "Topology", "EventType")
