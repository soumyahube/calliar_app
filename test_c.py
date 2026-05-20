# test_with_new_image.py
import pickle
import numpy as np
from services.cnn_encoder import CNNEncoderService
from services.knn_service import KNNTrajectoryService
import base64
from PIL import Image

print("="*60)
print("🧪 TEST AVEC UNE NOUVELLE IMAGE (HORS BASE)")
print("="*60)

# Charger les services
knn = KNNTrajectoryService(pkl_path='data/knn_calliar.pkl')
cnn = CNNEncoderService(model_path='data/arabic_encoder_calliar.pth')

# Chemin de l'image de test (celle que vous avez téléchargée)
test_image_path = 'data/2062.png'  # Mettez le bon chemin

# Lire et encoder l'image
with open(test_image_path, 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

embedding = cnn.encode_image(f"data:image/png;base64,{img_b64}")

# Recherche KNN
traj, confidence, best_match = knn.find_trajectory(embedding)

print(f"\n🔍 RÉSULTAT:")
print(f"   Mot trouvé: {best_match}")
print(f"   Confiance: {confidence:.4f}")

if confidence > 0.7:
    print("   ✅ Bon résultat !")
elif confidence > 0.5:
    print("   ⚠️ Résultat moyen")
else:
    print("   ❌ Mauvais résultat - problème de généralisation")