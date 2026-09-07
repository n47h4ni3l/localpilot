import unittest

from alerts.models import Alert
from alerts.render import render_alert
from alerts.severity import Severity


class ScopedMultifileAcceptance(unittest.TestCase):
    def test_default_preserves_two_argument_construction(self):
        alert = Alert("ready", "Ready")
        self.assertIs(alert.severity, Severity.info)
        self.assertEqual(render_alert(alert), "[INFO] Ready")

    def test_explicit_severity_is_rendered(self):
        alert = Alert("disk", "Disk nearly full", Severity.critical)
        self.assertEqual(render_alert(alert), "[CRITICAL] Disk nearly full")
        self.assertEqual(Severity.warning.value, "warning")


if __name__ == "__main__":
    unittest.main()
