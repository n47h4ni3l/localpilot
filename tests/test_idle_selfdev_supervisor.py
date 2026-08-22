#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from localpilot.idle_selfdev_supervisor import IdleSelfDevSupervisor
from localpilot.config import Config
from localpilot.safety import Safety


class TestIdleSelfDevSupervisor(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.safety = Safety()
        self.supervisor = IdleSelfDevSupervisor(self.config, self.safety)

    def test_pause_resume(self):
        # Simulate user activity
        self.supervisor.check_user_activity()
        self.supervisor.pause_selfdev()
        self.assertTrue(self.supervisor.user_activity_detected)

        # Simulate idle time
        self.supervisor.last_activity_time = time.time() - self.config.idle_threshold - 1
        self.supervisor.resume_selfdev()
        self.assertFalse(self.supervisor.user_activity_detected)

    def test_no_resume(self):
        # Simulate user activity
        self.supervisor.check_user_activity()
        self.supervisor.pause_selfdev()
        self.assertTrue(self.supervisor.user_activity_detected)

        # Simulate idle time less than threshold
        self.supervisor.last_activity_time = time.time() - self.config.idle_threshold + 1
        self.supervisor.resume_selfdev()
        self.assertTrue(self.supervisor.user_activity_detected)

if __name__ == '__main__':
    unittest.main()