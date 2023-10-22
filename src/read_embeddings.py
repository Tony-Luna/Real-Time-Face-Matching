#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 01:50:30 2023

@author: antonio
"""

import sqlite3
import numpy as np

# Connect to the SQLite database
conn = sqlite3.connect('embeddings/employee_embeddings_v2.db')
cur = conn.cursor()

# Execute a SELECT query to fetch all records from the 'employees' table
cur.execute("SELECT * FROM employees")
rows = cur.fetchall()

embeddings_list = []

# Iterate through the results and print the employee information and embeddings
for row in rows:
    employee_id, name, role, embedding_blob = row
    embedding = np.frombuffer(embedding_blob, dtype=np.float32)  # Convert blob to numpy array
    print(f"Employee ID: {employee_id}")
    print(f"Name: {name}")
    print(f"Role: {role}")
    print(f"Embedding: {embedding}\n")
    print(f"Embedding shape: {embedding.shape}")
    
    embeddings_list.append(embedding)

# Close the database connection
conn.close()