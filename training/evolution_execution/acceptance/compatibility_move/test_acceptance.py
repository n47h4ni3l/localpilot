import unittest
from pathlib import Path


class CompatibilityMoveAcceptance(unittest.TestCase):
    def test_old_import_is_same_function_object(self):
        from acme.legacy import normalize_tag as legacy_normalize
        from acme.text import normalize_tag as text_normalize

        self.assertIs(legacy_normalize, text_normalize)

    def test_slug_contract(self):
        from acme.slug import slug_for

        self.assertEqual(slug_for("  Hello World  "), "item-hello-world")
        self.assertEqual(slug_for("Blue Sky", prefix="Project Alpha"), "project-alpha-blue-sky")

    def test_candidate_added_focused_tests(self):
        content = Path("tests/test_text.py").read_text(encoding="utf-8")
        self.assertIn("normalize_tag", content)
        self.assertIn("slug_for", content)
        self.assertIn("assert", content)


if __name__ == "__main__":
    unittest.main()
