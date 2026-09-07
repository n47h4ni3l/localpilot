import unittest

from parser import parse_setting


class DiagnoseRepairAcceptance(unittest.TestCase):
    def test_comment_spacing_variants(self):
        self.assertEqual(parse_setting("retries=7#bounded"), ("retries", 7))
        self.assertEqual(parse_setting(" workers = 4   # local "), ("workers", 4))

    def test_plain_value_is_unchanged(self):
        self.assertEqual(parse_setting("timeout=30"), ("timeout", 30))


if __name__ == "__main__":
    unittest.main()
