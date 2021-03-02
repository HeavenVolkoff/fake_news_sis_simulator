# Internal
import unittest

# External
from simulador_sis_fakenews import EventType, Simulator


class TestImport(unittest.TestCase):
    def test_example_1(self) -> None:
        Simulator(
            ([EventType.Fake], *([EventType.Truth] for _ in range(99))),
            fake_rate_heuristic=lambda _: 1,
            truth_rate_heuristic=lambda _: 1,
            internal_fake_transmission_rate=0.5,
            external_fake_transmission_rate=0,
            internal_truth_transmission_rate=0.15,
            external_truth_transmission_rate=0,
        ).debug(300)


if __name__ == "__main__":
    unittest.main()
