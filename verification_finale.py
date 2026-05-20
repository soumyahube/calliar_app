# verification_finale.py
import pickle
import numpy as np
from services.cnn_encoder import CNNEncoderService
import base64
from PIL import Image

print("="*60)
print("🔍 VÉRIFICATION FINALE DE COHÉRENCE")
print("="*60)

# 1. Charger la base
with open('data/knn_calliar.pkl', 'rb') as f:
    knn_data = pickle.load(f)

print(f"\n📊 Base KNN:")
print(f"   train_embs shape: {knn_data['train_embs'].shape}")
print(f"   train_embs mean: {knn_data['train_embs'].mean():.6f}")
print(f"   train_embs std: {knn_data['train_embs'].std():.6f}")
print(f"   emb_mean shape: {knn_data['emb_mean'].shape}")
print(f"   emb_std shape: {knn_data['emb_std'].shape}")

# 2. Charger le modèle
cnn = CNNEncoderService(model_path='data/arabic_encoder_calliar_FINETUNED.pth')

# 3. Tester avec une image
test_path = 'data/14.png'
with open(test_path, 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

embedding = cnn.encode_image(f"data:image/png;base64,{img_b64}")
print(f"\n📸 Embedding généré depuis {test_path}:")
print(f"   mean: {embedding.mean():.6f}")
print(f"   std: {embedding.std():.6f}")
print(f"   norm: {np.linalg.norm(embedding):.6f}")

# 4. Appliquer la normalisation de la base
query_norm = (embedding - knn_data['emb_mean']) / knn_data['emb_std']
print(f"\n📐 Après normalisation Z-score:")
print(f"   mean: {query_norm.mean():.6f}")
print(f"   std: {query_norm.std():.6f}")

# 5. Vérifier que l'embedding normalisé est dans le même espace
from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(n_neighbors=1, metric='cosine')
knn.fit(knn_data['train_embs'])

distances, indices = knn.kneighbors(query_norm.reshape(1, -1))

print(f"\n🔍 RECHERCHE KNN:")
print(f"   Distance cosine: {distances[0][0]:.6f}")
print(f"   Confiance: {1.0 - distances[0][0]:.6f}")
print(f"   Match trouvé: {knn_data['train_texts'][indices[0][0]]}")

# 6. Vérification supplémentaire - la requête doit être proche d'un embedding de la base
similarities = np.dot(knn_data['train_embs'], query_norm)
print(f"\n📊 Similarités avec la base:")
print(f"   Max: {similarities.max():.6f}")
print(f"   Min: {similarities.min():.6f}")
print(f"   Mean: {similarities.mean():.6f}")

if similarities.max() > 0.9:
    print("\n✅ COHÉRENCE PARFAITE !")
    print("   Le modèle et la base KNN sont bien alignés.")
else:
    print("\n⚠️ Vérifiez que vous utilisez les bons fichiers")