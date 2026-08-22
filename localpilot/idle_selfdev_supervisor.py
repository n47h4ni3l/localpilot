#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import time
from localpilot.config import Config
from localpilot.safety import Safety


class IdleSelfDevSupervisor:
    def __init__(self, config: Config, safety: Safety):
        self.config = config
        self.safety = safety
        self.user_activity_detected = False
        self.last_activity_time = time.time()
        self.selfdev_task = None

    def check_user_activity(self):
        # Placeholder for user activity detection logic
        pass

    def pause_selfdev(self):
        if self.selfdev_task:
            self.selfdev_task.pause()
            self.user_activity_detected = True
            self.last_activity_time = time.time()

    def resume_selfdev(self):
        if self.user_activity_detected and time.time() - self.last_activity_time > self.config.idle_threshold:
            self.selfdev_task.resume()
            self.user_activity_detected = False

    def run(self):
        while True:
            self.check_user_activity()
            self.pause_selfdev()
            self.resume_selfdev()
            time.sleep(self.config.check_interval)


if __name__ == '__main__':
    config = Config()
    safety = Safety()
    supervisor = IdleSelfDevSupervisor(config, safety)
    supervisor.run()