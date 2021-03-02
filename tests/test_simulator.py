# Internal
import typing as T
import unittest
from collections import Counter

# External
from fake_news_sis_simulator import Event, EventType, Simulator, TimelineType

MAX_STEPS = 100


class TestImport(unittest.TestCase):
    def _test_simulator(self, *args: T.Any, **kwargs: T.Any) -> None:
        users = ([EventType.Fake], *([EventType.Genuine] for _ in range(99)))
        simulator = iter(Simulator(users, *args, **kwargs))

        for _ in range(MAX_STEPS):
            clock, event, stats = next(simulator)
            self.assertGreater(clock, 0)
            self.assertIsInstance(event, Event)
            self.assertIsInstance(stats, Counter)
            self.assertEqual(sum(stats.values()), len(users))

    def test_simulator_fifo(self) -> None:
        self._test_simulator(
            timeline_type=TimelineType.FIFO,
            fake_rate_heuristic=lambda _: 1,
            genuine_rate_heuristic=lambda _: 1,
            internal_fake_transmission_rate=0.5,
            external_fake_transmission_rate=0,
            internal_genuine_transmission_rate=0.15,
            external_genuine_transmission_rate=0,
        )

    def test_simulator_rnd(self) -> None:
        self._test_simulator(
            timeline_type=TimelineType.RND,
            fake_rate_heuristic=lambda _: 1,
            genuine_rate_heuristic=lambda _: 1,
            internal_fake_transmission_rate=0.5,
            external_fake_transmission_rate=0,
            internal_genuine_transmission_rate=0.15,
            external_genuine_transmission_rate=0,
        )


if __name__ == "__main__":
    unittest.main()
