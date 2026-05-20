# app.py
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
from flask import Flask, request, jsonify, render_template
import numpy as np
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.knn_service import KNNTrajectoryService
from services.cnn_encoder import CNNEncoderService

app = Flask(__name__)

# ============================
# CHEMINS
# ============================
DATA_PATH  = os.path.join(os.path.dirname(__file__), 'data/knn_calliar_Flask.pkl')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data/arabic_encoder_calliar_FINETUNED.pth')
# MODEL_PATH peut être None si tu n'as pas encore exporté les poids depuis Colab

# ============================
# CHARGEMENT DES SERVICES
# ============================
if os.path.exists(DATA_PATH):
    knn_service = KNNTrajectoryService(pkl_path=DATA_PATH)
else:
    print(f"❌ Fichier KNN non trouvé: {DATA_PATH}")
    knn_service = None


cnn_service = CNNEncoderService(
    model_path=MODEL_PATH,  
    embedding_dim=128
)
# ============================
# CONVERSION TRAJECTOIRE → STROKES
# ============================
def traj_to_strokes(traj, threshold=0.05):
    """Sépare la trajectoire en strokes pour l'affichage."""
    strokes = []
    current = [[float(traj[0][0]), float(traj[0][1])]]

    for i in range(1, len(traj)):
        dx = traj[i][0] - traj[i-1][0]
        dy = traj[i][1] - traj[i-1][1]
        dist = (dx**2 + dy**2) ** 0.5

        if dist > threshold:
            if len(current) > 1:
                strokes.append(current)
            current = [[float(traj[i][0]), float(traj[i][1])]]
        else:
            current.append([float(traj[i][0]), float(traj[i][1])])

    if len(current) > 1:
        strokes.append(current)

    return strokes if strokes else [[[float(p[0]), float(p[1])] for p in traj]]

# ============================
# ROUTES
# ============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'knn_loaded': knn_service is not None,
        'cnn_loaded': True
    })

@app.route('/predict', methods=['POST'])
def predict():
    if knn_service is None:
        return jsonify({'error': 'Service KNN non disponible'}), 500

    try:
        start_time = time.time()
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'error': 'Aucune image fournie'}), 400

        embedding = cnn_service.encode_image(data['image'])
        
        # Récupérer trajectoire
        trajectory, confidence, best_match = knn_service.find_trajectory(embedding)
        
        # ============================================================
        # DEBUG: Afficher les informations de la trajectoire
        # ============================================================
        print("\n" + "="*50)
        print("🔍 DEBUG TRAJECTOIRE")
        print("="*50)
        print(f"Mot trouvé: {best_match}")
        print(f"Confiance: {confidence:.4f}")
        print(f"Shape trajectoire: {trajectory.shape}")
        print(f"Type: {type(trajectory)}")
        print(f"X - min: {trajectory[:,0].min():.4f}, max: {trajectory[:,0].max():.4f}")
        print(f"Y - min: {trajectory[:,1].min():.4f}, max: {trajectory[:,1].max():.4f}")
        print(f"Premier point: ({trajectory[0][0]:.4f}, {trajectory[0][1]:.4f})")
        print(f"Dernier point: ({trajectory[-1][0]:.4f}, {trajectory[-1][1]:.4f})")
        
        strokes = traj_to_strokes(trajectory)
        print(f"Nombre de strokes: {len(strokes)}")
        print("="*50 + "\n")
        # ============================================================
        
        return jsonify({
            'strokes': strokes,
            'points': trajectory.tolist(),
            'best_match': best_match,
            'confidence': round(float(confidence), 4),
            'time': round(time.time() - start_time, 3)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# Ajoutez cette route dans app.py
@app.route('/list_train_images', methods=['GET'])
def list_train_images():
    """Liste toutes les images d'entraînement disponibles"""
    try:
        if knn_service is None:
            return jsonify({'error': 'Service KNN non disponible'}), 500
        
        # Récupérer les textes (mots) disponibles
        train_texts = knn_service.train_texts
        
        # Compter les occurrences
        unique_texts = {}
        for text in train_texts:
            unique_texts[text] = unique_texts.get(text, 0) + 1
        
        # Trier par fréquence
        sorted_texts = sorted(unique_texts.items(), key=lambda x: x[1], reverse=True)
        
        return jsonify({
            'total_examples': len(train_texts),
            'unique_words': len(unique_texts),
            'examples': [
                {'word': word, 'count': count} 
                for word, count in sorted_texts[:50]  # Les 50 premiers
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search_word/<word>', methods=['GET'])
def search_word(word):
    """Recherche un mot spécifique dans la base"""
    try:
        if knn_service is None:
            return jsonify({'error': 'Service KNN non disponible'}), 500
        
        # Chercher le mot
        matches = []
        for i, text in enumerate(knn_service.train_texts):
            if word.lower() in text.lower():
                matches.append({
                    'index': i,
                    'text': text,
                    'trajectory_shape': knn_service.train_trajs[i].shape
                })
        
        return jsonify({
            'search_word': word,
            'found': len(matches),
            'matches': matches[:20]  # Limiter à 20 résultats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Ajoutez cette route dans app.py
@app.route('/debug_traj/<word>', methods=['GET'])
def debug_traj_word(word):
    """Test une trajectoire pour un mot spécifique"""
    try:
        # Chercher le mot dans la base
        matches = []
        for i, text in enumerate(knn_service.train_texts):
            if word in text:
                matches.append(i)
        
        if not matches:
            mots_exemples = list(set(knn_service.train_texts[:20]))
            return f"Mot '{word}' non trouvé. Exemples: {', '.join(mots_exemples[:10])}"
        
        # Prendre le premier match
        idx = matches[0]
        traj = knn_service.train_trajs[idx]
        text = knn_service.train_texts[idx]
        
        import base64
        import matplotlib.pyplot as plt
        from io import BytesIO
        
        plt.figure(figsize=(8, 8))
        plt.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2)
        plt.scatter(traj[0, 0], traj[0, 1], c='green', s=100, label='Début')
        plt.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, label='Fin')
        plt.xlim(-0.05, 1.05)
        plt.ylim(1.05, -0.05)
        plt.title(f"Mot: {text}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        
        return f'''
        <html>
        <head><title>Debug: {text}</title>
        <style>
            body {{ font-family: Arial; text-align: center; padding: 20px; }}
            img {{ border: 1px solid #ccc; margin: 20px; }}
            .info {{ background: #f0f0f0; padding: 10px; margin: 10px; border-radius: 5px; }}
        </style>
        </head>
        <body>
            <h1>Trajectoire pour: <span style="color:blue">{text}</span></h1>
            <img src="data:image/png;base64,{img_base64}" />
            <div class="info">
                <p><strong>Shape:</strong> {traj.shape}</p>
                <p><strong>X range:</strong> [{traj[:,0].min():.3f}, {traj[:,0].max():.3f}]</p>
                <p><strong>Y range:</strong> [{traj[:,1].min():.3f}, {traj[:,1].max():.3f}]</p>
            </div>
            <p><a href="/">Retour à l'application</a></p>
        </body>
        </html>
        '''
    except Exception as e:
        return f"Erreur: {e}"      
# Dans app.py, ajoutez :
@app.route('/test_embedding', methods=['POST'])
def test_embedding():
    """Teste si l'embedding généré est cohérent"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image'}), 400
        
        embedding = cnn_service.encode_image(data['image'])
        
        # Vérifier les statistiques
        return jsonify({
            'embedding_shape': len(embedding),
            'embedding_mean': float(embedding.mean()),
            'embedding_std': float(embedding.std()),
            'embedding_norm': float(np.linalg.norm(embedding))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# ============================
# LANCEMENT
# ============================
if __name__ == '__main__':
    print("=" * 50)
    print("🚀 Application Calliar - KNN + CNN")
    print("=" * 50)
    print(f"📁 KNN : {DATA_PATH}")
    print(f"📁 CNN : {MODEL_PATH}")
    print(f"✅ KNN chargé : {'oui' if knn_service else 'NON'}")
    print("🌐 http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=5000)