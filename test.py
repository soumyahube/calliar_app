# test_local_model.py
import torch
import os
import sys

# Ajouter le chemin du projet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. Vérifier que le fichier existe
model_path = 'data/arabic_encoder_calliar.pth'

print("="*60)
print("🔍 TEST DU MODÈLE LOCAL")
print("="*60)

if os.path.exists(model_path):
    size = os.path.getsize(model_path) / 1024
    print(f"✅ Fichier trouvé: {model_path}")
    print(f"   Taille: {size:.1f} KB")
    
    # Charger les poids
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        print(f"   Nombre de couches: {len(state_dict)}")
        
        # Afficher les stats des premières couches
        first_key = list(state_dict.keys())[0]
        print(f"   Première couche: {first_key}")
        print(f"   Moyenne des poids: {state_dict[first_key].mean():.6f}")
        print(f"   Écart-type: {state_dict[first_key].std():.6f}")
        
        print("\n✅ Modèle chargé avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
else:
    print(f"❌ Modèle non trouvé: {model_path}")
    print("   Vérifiez que le fichier existe dans le dossier data/")