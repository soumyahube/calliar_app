# check_knn.py
import pickle

with open('data/knn_calliar_Flask.pkl', 'rb') as f:
    data = pickle.load(f)
    print("✅ Fichier KNN chargé")
    print(f"   Clés: {data.keys()}")
    print(f"   train_embs shape: {data['train_embs'].shape}")
    print(f"   train_trajs shape: {data['train_trajs'].shape}")
    print(f"   train_texts: {len(data['train_texts'])} exemples")