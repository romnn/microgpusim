import unittest
from gpucachesim.plot import human_format_thousands


class TestHumanFormatThousands(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(human_format_thousands(1000.0, round_to=2), "1.00K")
        self.assertEqual(
            human_format_thousands(320.004, round_to=2),
            "320.00",
        )

    def test_variable_precision(self):
        self.assertEqual(
            human_format_thousands(320.0, round_to=2, variable_precision=True), "320"
        )
        self.assertEqual(
            human_format_thousands(320.2, round_to=2, variable_precision=True), "320.2"
        )
        self.assertEqual(
            human_format_thousands(320.02, round_to=2, variable_precision=True),
            "320.02",
        )
        self.assertEqual(
            human_format_thousands(320.004, round_to=2, variable_precision=True),
            "320",
        )
        self.assertEqual(
            human_format_thousands(1500, round_to=4, variable_precision=True),
            "1.5K",
        )
        self.assertEqual(
            human_format_thousands(1010.2, round_to=4, variable_precision=True),
            "1.0102K",
        )
        self.assertEqual(
            human_format_thousands(1010.0, round_to=4, variable_precision=True),
            "1.01K",
        )


if __name__ == "__main__":
    unittest.main()
