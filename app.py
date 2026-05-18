# calliar_app/app.py
from flask import Flask, request, jsonify, render_template
import numpy as np
import base64
import time
import os
import sys

# Ajouter le chemin pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.knn_service import KNNTrajectoryService

app = Flask(__name__)

# ============================
# CHARGEMENT DU SERVICE KNN
# ============================
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data/knn_calliar_k1.pkl')

if os.path.exists(DATA_PATH):
    knn_service = KNNTrajectoryService(pkl_path=DATA_PATH)
else:
    print(f"❌ Fichier non trouvé: {DATA_PATH}")
    knn_service = None

# ============================
# CONVERSION TRAJECTOIRE → STROKES
# ============================
def traj_to_strokes(traj, threshold=0.05):
    """Sépare la trajectoire en strokes pour l'animation"""
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
    return jsonify({'status': 'ok', 'knn_loaded': knn_service is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if knn_service is None:
        return jsonify({'error': 'Service KNN non disponible'}), 500
    
    try:
        start_time = time.time()
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'Aucune image'}), 400
        
        # Décoder l'image base64
        image_b64 = data['image']
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        
        # ============================================================
        # ICI : Tu dois remplacer par ton vrai CNN
        # Pour l'instant, embedding aléatoire (test)
        # ============================================================
        # TODO: Intégrer ton CNN ici
        embedding = np.random.randn(128).astype(np.float32)
        # ============================================================
        
        # Recherche KNN
        trajectory, confidence, best_match = knn_service.find_trajectory(embedding)
        strokes = traj_to_strokes(trajectory)
        
        return jsonify({
            'strokes': strokes,
            'points': trajectory.tolist(),
            'best_match': best_match,
            'confidence': round(confidence, 4),
            'time': round(time.time() - start_time, 3)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================
# LANCEMENT
# ============================
if __name__ == '__main__':
    print("="*50)
    print("🚀 Application Calliar - KNN Trajectory")
    print("="*50)
    print(f"📁 Fichier KNN: {DATA_PATH}")
    print(f"✅ Service KNN: {'chargé' if knn_service else 'NON DISPONIBLE'}")
    print("🌐 http://localhost:5000")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)