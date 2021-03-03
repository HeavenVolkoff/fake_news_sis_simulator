from __future__ import annotations

# Internal
import pickle
import typing as T
from enum import IntEnum, auto, unique
from heapq import heappop, heappush
from base64 import b64decode, b64encode
from collections import Counter

# External
import numpy as np
from numpy.random import Generator, default_rng

if T.TYPE_CHECKING:
    # External
    from rich.pretty import Pretty


@unique
class EventType(IntEnum):
    Fake = 0
    Genuine = 1

    @staticmethod
    def fake_counter(timeline: T.Sequence[EventType]) -> int:
        return len(timeline) - sum(timeline)

    @staticmethod
    def genuine_counter(timeline: T.Sequence[EventType]) -> int:
        return sum(timeline)


@unique
class EventOrigin(IntEnum):
    External = auto()
    Internal = auto()


@unique
class TopologyType(IntEnum):
    Star = auto()
    Clique = auto()


@unique
class TimelineType(IntEnum):
    RND = auto()
    FIFO = auto()


class Event(T.NamedTuple):
    type: EventType
    origin: EventOrigin
    user_id: int


class Simulator(T.Iterable[T.Tuple[float, Event, T.Counter[T.Tuple[EventType, ...]]]]):
    """SIS epidemic model simulator for fake news propagation."""

    def __init__(
        self,
        users: T.Sequence[T.MutableSequence[EventType]],
        *,
        timeline_type: TimelineType = TimelineType.FIFO,
        topology_type: TopologyType = TopologyType.Clique,
        fake_rate_heuristic: T.Callable[[int], T.Union[int, float]],
        genuine_rate_heuristic: T.Callable[[int], T.Union[int, float]],
        internal_fake_transmission_rate: float,
        external_fake_transmission_rate: float,
        internal_genuine_transmission_rate: float,
        external_genuine_transmission_rate: float,
    ) -> None:
        self._iter: T.Optional[
            T.Iterator[T.Tuple[float, Event, T.Counter[T.Tuple[EventType, ...]]]]
        ] = None
        self._seed: T.Optional[str] = None

        self.rng = default_rng(None)
        self.clock: float = 0
        self.topology = topology_type
        self.iteration = 0
        self.timeline_type = timeline_type
        self.users_timeline = users
        self.fake_rate_heuristic = fake_rate_heuristic
        self.genuine_rate_heuristic = genuine_rate_heuristic
        self.external_fake_transmission_rate = external_fake_transmission_rate
        self.internal_fake_transmission_rate = internal_fake_transmission_rate
        self.internal_genuine_transmission_rate = internal_genuine_transmission_rate
        self.external_genuine_transmission_rate = external_genuine_transmission_rate

    def __iter__(self) -> T.Iterator[T.Tuple[float, Event, T.Counter[T.Tuple[EventType, ...]]]]:
        if self._iter is None:
            self._iter = iter(self.step())
        return self._iter

    def __rich__(self) -> "Pretty":
        # External
        from rich.pretty import Pretty

        return Pretty(
            {
                "users": len(self.users_timeline),
                "clock": self.clock,
                "iteration": self.iteration,
                "timeline": self.timeline_type.name,
                "topology": self.topology.name,
                "rate": {
                    "internal": {
                        "fake": self.internal_fake_transmission_rate,
                        "genuine": self.internal_genuine_transmission_rate,
                    },
                    "external": {
                        "fake": self.external_fake_transmission_rate,
                        "genuine": self.external_genuine_transmission_rate,
                    },
                },
            }
        )

    @property
    def seed(self) -> str:
        """Generate seed from current random generator"""
        if self._seed is None:
            self._seed = b64encode(pickle.dumps(self.rng)).decode(encoding="utf8")
        return self._seed

    def step(
        self,
    ) -> T.Generator[T.Tuple[float, Event, T.Counter[T.Tuple[EventType, ...]]], None, None]:
        """Run a single step of the Simulator."""
        while True:
            event_queue: T.List[T.Tuple[float, Event]] = []
            self.gen_events(event_queue)
            if len(event_queue) == 0:
                break

            delta, event = heappop(event_queue)

            timeline = self.users_timeline[event.user_id]
            if self.timeline_type == TimelineType.RND:
                timeline[self.rng.integers(0, len(timeline))] = event.type
            elif self.timeline_type == TimelineType.FIFO:
                timeline.insert(0, event.type)
                timeline.pop()
            else:
                raise ValueError("Timeline type must be a valid enum member")

            stats: T.Counter[T.Tuple[EventType, ...]] = Counter()
            for timeline in self.users_timeline:
                stats[tuple(timeline)] += 1

            # Update clock
            self.clock += delta
            self.iteration += 1
            yield self.clock, event, stats

    def load_seed(self, seed: str) -> None:
        """Load a seed to reenact a previous simulator run.

        Arguments:
            seed: Base64 encode of a pickled numpy.random.Generator instance

        """
        rng = pickle.loads(b64decode(seed))

        if not isinstance(rng, Generator):
            raise ValueError("Seed must be a pickle of a numpy.random.Generator")

        self._seed = seed
        self.rng = rng

    def gen_events(self, event_queue: T.List[T.Tuple[float, Event]]) -> None:
        """Populate event queue with a round of generated events.

        Argument
            event_queue: Event queue to be populated.

        """
        for event_type, rate in (
            (EventType.Fake, self.external_fake_transmission_rate),
            (EventType.Genuine, self.external_genuine_transmission_rate),
        ):
            if rate <= 0:
                continue
            external_events = [
                Event(
                    event_type,
                    EventOrigin.External,
                    timeline_id,
                )
                for timeline_id in range(len(self.users_timeline))
            ]
            for entry in zip(
                self.rng.exponential(
                    # Sample an exponential distribution with scale parameter of λ = 1/β to
                    # retrieve the time this event will happen
                    1 / rate,
                    len(external_events),
                ),
                external_events,
            ):
                heappush(event_queue, entry)

        for event_type, heuristic, rate, user_counter in (
            (
                EventType.Fake,
                self.fake_rate_heuristic,
                self.internal_fake_transmission_rate,
                EventType.fake_counter,
            ),
            (
                EventType.Genuine,
                self.genuine_rate_heuristic,
                self.internal_genuine_transmission_rate,
                EventType.genuine_counter,
            ),
        ):
            if rate <= 0:
                continue

            internal_events = [
                (
                    user_counter(neighbour),
                    Event(
                        event_type,
                        EventOrigin.Internal,
                        neighbour_id,
                    ),
                )
                # Loop through all infected users
                for current_id, user in enumerate(self.users_timeline)
                if user_counter(user) > 0
                # Loop through all neighbor
                for neighbour_id, neighbour in enumerate(self.users_timeline)
                if neighbour_id != current_id
            ]
            for time, (_, event) in zip(
                self.rng.exponential(
                    # Sample an exponential distribution with scale parameter of λ = 1/β to
                    # retrieve the time this event will happen
                    1
                    / (
                        # β = f₁(k)μ₁
                        # Represents the rate which a user that has (k) fake news in its
                        # timeline share on with its neighbours
                        np.array([heuristic(count) for count, _ in internal_events])
                        * rate
                    ),
                ),
                internal_events,
            ):
                heappush(event_queue, (time, event))


__all__ = ("Event", "TopologyType", "EventType", "Simulator", "EventOrigin", "TimelineType")
