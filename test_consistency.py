# test_consistency.py - Version avec une vraie image
import numpy as np
import pickle
import sys
import os
from PIL import Image
import base64
import io

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services.cnn_encoder import CNNEncoderService

def test_with_real_image():
    """Test avec une vraie image du dataset Calliar"""
    
    # 1. Charger la base KNN
    with open('data/knn_calliar.pkl', 'rb') as f:
        knn_data = pickle.load(f)
    
    # 2. Charger le modèle CNN
    cnn_service = CNNEncoderService(model_path='data/arabic_encoder_calliar.pth')
    
    # 3. PRENEZ UNE VRAIE IMAGE de votre dossier data/
    #    Par exemple, si vous avez une image test dans data/test.png
    image_path = 'data/14.png'  # ← METTEZ LE CHEMIN D'UNE VRAIE IMAGE
    
    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée: {image_path}")
        print("   Veuillez mettre une image dans le dossier data/")
        return
    
    # Charger et convertir l'image en base64
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode()
    
    # Générer l'embedding
    embedding = cnn_service.encode_image(f"data:image/png;base64,{img_b64}")
    
    print(f"✅ Embedding généré depuis: {image_path}")
    print(f"   Moyenne: {embedding.mean():.6f}")
    print(f"   Norme: {np.linalg.norm(embedding):.6f}")
    
    # 4. Comparer avec la base KNN
    query_norm = (embedding - knn_data['emb_mean']) / knn_data['emb_std']
    
    # Trouver le voisin le plus proche
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=1, metric='cosine')
    knn.fit(knn_data['train_embs'])
    
    distances, indices = knn.kneighbors(query_norm.reshape(1, -1))
    best_match = knn_data['train_texts'][indices[0][0]]
    confidence = 1.0 - distances[0][0]
    
    print(f"\n🔍 Résultat de la recherche:")
    print(f"   Mot trouvé: {best_match}")
    print(f"   Confiance: {confidence:.4f}")
    print(f"   Distance: {distances[0][0]:.4f}")

if __name__ == "__main__":
    test_with_real_image()