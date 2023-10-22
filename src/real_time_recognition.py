#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 01:55:37 2023

@author: antonio
"""

import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors
import torch  # Import torch to resolve a potential error in unsqueeze

# Initialize models
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load embeddings from database
conn = sqlite3.connect('embeddings/employee_embeddings.db')  # Updated path to DB
cur = conn.cursor()
cur.execute("SELECT id, name, role, embedding FROM employees")
db_data = cur.fetchall()
ids, names, roles, db_embeddings = zip(*db_data)
db_embeddings = [np.frombuffer(e, dtype=np.float32) for e in db_embeddings]

# Function to match face based on the chosen approach
def match_face(approach, img_embedding):
    try:
        if approach == 'cosine':
            similarity = cosine_similarity(img_embedding.detach().numpy(), np.array(db_embeddings))
            most_similar_index = np.argmax(similarity)
            if similarity[0, most_similar_index] > 0.75:
                return f"{names[most_similar_index]} - {roles[most_similar_index]}"
        elif approach == 'euclidean':
            img_embedding_1d = np.squeeze(img_embedding.detach().numpy())
            distances = [euclidean(img_embedding_1d, np.squeeze(db_embedding)) for db_embedding in db_embeddings]
            
            print(distances)
            
            most_similar_index = np.argmin(distances)
            if distances[most_similar_index] < 10:  # Example threshold, adjust as needed
                return f"{names[most_similar_index]} - {roles[most_similar_index]}"
        elif approach == 'knn':
            knn = NearestNeighbors(n_neighbors=3)  # For example, 3-nearest neighbors
            knn.fit(np.array(db_embeddings))
            distances, indices = knn.kneighbors(img_embedding.detach().numpy())
            # Here, we'll just take the closest neighbor, but you could also vote among top k neighbors
            most_similar_index = indices[0][0]
            if distances[0][0] < 10:  # Example threshold, adjust as needed
                return f"{names[most_similar_index]} - {roles[most_similar_index]}"
    except Exception as e:
        print(f"Error: {e}")
        return "?"  # Return ? if an error occurs or no match found
    return "?"  # Return ? if no match found

# Start video capture
cap = cv2.VideoCapture(0)
matching_approach = 'euclidean'  # Choose from 'cosine', 'euclidean', or 'knn'

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        for box in boxes.astype(int):
            face = frame[box[1]:box[3], box[0]:box[2]]
            if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
                continue
            img_cropped = cv2.resize(face, (160, 160))
            img_embedding = resnet(torch.tensor(img_cropped).permute(2, 0, 1).unsqueeze(0).float())
            info = match_face(matching_approach, img_embedding)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, info, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
