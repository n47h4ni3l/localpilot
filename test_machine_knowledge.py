#!/usr/bin/env python3
import unittest
import sqlite3
import machine_knowledge

# Connect to the SQLite database
conn = sqlite3.connect('machine_knowledge.db')
cur = conn.cursor()

# Test case for storing a machine fact
# This test case will store a fact and then retrieve it to ensure it was stored correctly

class TestMachineKnowledge(unittest.TestCase):
    def test_store_fact(self):
        # Store a fact
        machine_knowledge.store_fact('os', 'Windows')

        # Retrieve the fact
        cur.execute("SELECT * FROM facts WHERE key = 'os'")
        fact = cur.fetchone()

        # Assert that the fact was stored correctly
        self.assertIsNotNone(fact)
        self.assertEqual(fact[1], 'os')
        self.assertEqual(fact[2], 'Windows')

    # Test case for recording a change in machine fact
    # This test case will record a change and then retrieve it to ensure it was recorded correctly
    def test_record_change(self):
        # Record a change
        machine_knowledge.record_change('os', 'Windows', 'Linux')

        # Retrieve the change
        cur.execute("SELECT * FROM history WHERE key = 'os'")
        change = cur.fetchone()

        # Assert that the change was recorded correctly
        self.assertIsNotNone(change)
        self.assertEqual(change[1], 'os')
        self.assertEqual(change[2], 'Windows')
        self.assertEqual(change[3], 'Linux')
        self.assertEqual(change[4], 'script')

if __name__ == '__main__':
    unittest.main()