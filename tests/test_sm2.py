import unittest
from datetime import timezone

from src.repetition.sm2 import calculate_sm2


class CalculateSm2Tests(unittest.TestCase):
    def test_successive_correct_answers_grow_review_intervals(self) -> None:
        first = calculate_sm2(5, 0, 0, 2.5)
        second = calculate_sm2(5, first[0], first[1], first[2])
        third = calculate_sm2(5, second[0], second[1], second[2])

        self.assertEqual((first[0], first[1]), (1, 1))
        self.assertEqual(first[3].tzinfo, timezone.utc)
        self.assertEqual((second[0], second[1]), (2, 6))
        self.assertEqual((third[0], third[1]), (3, 16))

    def test_low_rating_resets_repetition_count(self) -> None:
        repetition_count, interval_days, ease_factor, _ = calculate_sm2(
            1, 4, 20, 2.5
        )

        self.assertEqual(repetition_count, 0)
        self.assertEqual(interval_days, 1)
        self.assertGreaterEqual(ease_factor, 1.3)

    def test_invalid_rating_and_state_are_rejected(self) -> None:
        invalid_states = [
            (6, 0, 0, 2.5),
            (3, -1, 0, 2.5),
            (3, 1, -1, 2.5),
            (3, 1, 1, float("nan")),
            (3, 1, 1, 1.2),
        ]

        for values in invalid_states:
            with self.subTest(values=values), self.assertRaises(ValueError):
                calculate_sm2(*values)


if __name__ == "__main__":
    unittest.main()
