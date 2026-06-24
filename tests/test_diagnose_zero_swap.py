import unittest

from tools.diagnose_zero_swap import classify_zero_swap, quantiles


class DiagnoseZeroSwapTest(unittest.TestCase):
    def test_classifies_blur_only_success(self):
        self.assertEqual(
            classify_zero_swap(
                {
                    "non_target_face_rows": 10,
                    "non_target_swap_rows": 0,
                    "non_target_blur_rows": 10,
                    "non_target_preserve_rows": 0,
                    "non_target_unprocessed_rows": 0,
                    "non_target_unknown_rows": 0,
                }
            ),
            "blur_only_success",
        )

    def test_classifies_privacy_exposure(self):
        self.assertEqual(
            classify_zero_swap(
                {
                    "non_target_face_rows": 10,
                    "non_target_swap_rows": 0,
                    "non_target_blur_rows": 3,
                    "non_target_preserve_rows": 1,
                    "non_target_unprocessed_rows": 0,
                    "non_target_unknown_rows": 6,
                }
            ),
            "privacy_exposure",
        )

    def test_quantiles(self):
        self.assertEqual(
            quantiles([10, 20, 30, 40, 50]),
            {"min": 10, "p25": 20, "median": 30, "p75": 40, "max": 50},
        )
        self.assertEqual(quantiles([]), {})


if __name__ == "__main__":
    unittest.main()
