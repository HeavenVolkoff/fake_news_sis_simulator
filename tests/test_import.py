# Internal
import unittest


class TestImport(unittest.TestCase):
    def test_import(self) -> None:
        # External
        from fake_news_sis_simulator import (
            Event,
            EventType,
            Simulator,
            EventOrigin,
            TimelineType,
            TopologyType,
        )


if __name__ == "__main__":
    unittest.main()
