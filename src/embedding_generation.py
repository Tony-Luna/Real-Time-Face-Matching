#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 18:24:31 2023

@author: antonio
"""

import cv2
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
import sqlite3
from tqdm import tqdm
import torch  # Import torch to resolve an error in unsqueeze

# Initialize models
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load employee data
employee_data = pd.read_csv('data/employee_data.csv')  # Updated path to CSV

# SQLite Database setup
conn = sqlite3.connect('embeddings/employee_embeddings.db')  # Updated path to DB
cur = conn.cursor()
cur.execute('''
            CREATE TABLE IF NOT EXISTS employees (
            id INT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            embedding BLOB NOT NULL)
            ''')
conn.commit()

# Process each employee's image
for index, row in tqdm(employee_data.iterrows(), total=len(employee_data)):
    image_path = row['Face image path']
    image = cv2.imread(image_path)
    boxes, _ = mtcnn.detect(image)
    if boxes is not None:
        box = boxes[0].astype(int)
        face = image[box[1]:box[3], box[0]:box[2]]
        img_cropped = cv2.resize(face, (160, 160))
        img_embedding = resnet(torch.tensor(img_cropped).permute(2, 0, 1).unsqueeze(0).float())  # Updated embedding line
        embedding_blob = img_embedding.detach().numpy().tobytes()
        cur.execute("INSERT INTO employees (id, name, role, embedding) VALUES (?, ?, ?, ?)",
                    (row['Employee id'], row['Name'], row['Role'], embedding_blob))

conn.commit()
conn.close()