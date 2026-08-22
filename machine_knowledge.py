#!/usr/bin/env python3
import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('machine_knowledge.db')
cur = conn.cursor()

# Create tables if they don't exist
create_facts_table = '''
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
'''
create_history_table = '''
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY,
    key TEXT NOT NULL,
    old_value TEXT NOT NULL,
    new_value TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    source TEXT NOT NULL
);
'''

cur.execute(create_facts_table)
cur.execute(create_history_table)

# Function to store a machine fact
def store_fact(key, value):
    cur.execute("INSERT INTO facts (key, value) VALUES (?, ?)", (key, value))
    conn.commit()

# Function to record a change in machine fact
# This function should be called whenever a fact is updated
# It records the old and new values along with the timestamp and source of the change
# The source could be a script name, user input, or any other identifier
# For simplicity, we will use 'script' as the source here
# In a real-world scenario, you would pass the source as an argument
# and store it in the database

def record_change(key, old_value, new_value):
    cur.execute("INSERT INTO history (key, old_value, new_value, source) VALUES (?, ?, ?, ?)", (key, old_value, new_value, 'script'))
    conn.commit()

# Example usage
store_fact('os', 'Windows')
record_change('os', 'Windows', 'Linux')

# Close the connection
conn.close()