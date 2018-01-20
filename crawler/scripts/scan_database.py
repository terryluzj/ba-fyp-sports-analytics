import os
import sqlite3

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
database_path = os.path.join(parent_path, 'data/race.db')

# Establish database connection
connection = sqlite3.connect(database_path)
cursor = connection.cursor()

data = cursor.execute('''
    SELECT *
    FROM individual_record
''').fetchall()

for element in data:
    print(element)
