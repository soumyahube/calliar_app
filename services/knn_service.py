# services/knn_service.py
import numpy as np
import pickle
import os

class KNNTrajectoryService:
    def __init__(self, pkl_path='data/knn_calliar_Flask.pkl'):
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Fichier non trouvé: {pkl_path}")
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.train_embs = data['train_embs']
        self.train_trajs = data['train_trajs']
        self.train_texts = data['train_texts']
        self.emb_mean = data['emb_mean']
        self.emb_std = data['emb_std']
        
        # Vérifier les trajectoires
        print(f"✅ KNN chargé: {len(self.train_embs)} exemples")
        print(f"   Shape des trajectoires: {self.train_trajs.shape}")
        print(f"   Exemple trajectoire - min: {self.train_trajs[0].min():.4f}, max: {self.train_trajs[0].max():.4f}")
        
        from sklearn.neighbors import NearestNeighbors
        self.knn = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.knn.fit(self.train_embs)
    
    def find_trajectory(self, embedding):
        q = ((embedding - self.emb_mean) / self.emb_std).reshape(1, -1)
        distances, indices = self.knn.kneighbors(q)
        
        idx = indices[0][0]
        trajectory = self.train_trajs[idx].copy()
        confidence = float(1.0 - distances[0][0])
        best_match = self.train_texts[idx]
        
        return trajectory, confidence, best_match