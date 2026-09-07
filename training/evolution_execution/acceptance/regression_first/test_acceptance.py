import unittest
import ast
from pathlib import Path

from pricing import bulk_total


class RegressionFirstAcceptance(unittest.TestCase):
    def test_boundary_receives_discount(self):
        self.assertEqual(bulk_total(10, 10.0), 90.0)

    def test_candidate_added_boundary_regression(self):
        content = Path("tests/test_pricing.py").read_text(encoding="utf-8")
        tree = ast.parse(content)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        boundary_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "bulk_total"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 10
        ]
        self.assertTrue(boundary_calls, "tests/test_pricing.py needs a quantity-10 regression")


if __name__ == "__main__":
    unittest.main()
