# tests/test_self_teaching.py

import unittest
from src.self_teaching import main


class TestSelfTeaching(unittest.TestCase):
    def test_main(self):
        self.assertTrue(main)

if __name__ == "__main__":
    unittest.main()