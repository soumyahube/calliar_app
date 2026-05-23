# services/ocr_service.py
import easyocr
import numpy as np
from PIL import Image
import base64
import io

class ArabicOCRService:
    def __init__(self):
        print("⏳ Chargement EasyOCR arabe...")
        self.reader = easyocr.Reader(['ar'], gpu=False)
        print("✅ EasyOCR prêt")

    def read_word(self, image_data):
      if ',' in image_data:
        image_data = image_data.split(',')[1]
      image_bytes = base64.b64decode(image_data)
      img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
      img_np = np.array(img)

      print(f"\n📸 Image reçue : {img.size}, mode={img.mode}")

    # Paramètres optimisés pour dessin manuscrit
      results_normal = self.reader.readtext(
        img_np,
        detail=1,
        paragraph=False,
        width_ths=0.9,      # tolérance largeur
        height_ths=0.9,     # tolérance hauteur
        min_size=20,        # ignorer les très petits éléments
        text_threshold=0.3, # seuil bas pour capturer l'écriture manuscrite
        low_text=0.25,
        link_threshold=0.2
      )

      img_inverted = 255 - img_np
      results_inverted = self.reader.readtext(
        img_inverted,
        detail=1,
        paragraph=False,
        width_ths=0.9,
        height_ths=0.9,
        min_size=20,
        text_threshold=0.3,
        low_text=0.25,
        link_threshold=0.2
      )

      print(f"🔍 Détections (normale)  : {len(results_normal)}")
      for r in results_normal:
        print(f"   '{r[1]}'  conf={r[2]:.3f}")

      print(f"🔍 Détections (inversée) : {len(results_inverted)}")
      for r in results_inverted:
        print(f"   '{r[1]}'  conf={r[2]:.3f}")

      all_results = results_normal + results_inverted

      if not all_results:
        print("⚠️  Aucun texte détecté")
        return None, 0.0

      best  = max(all_results, key=lambda x: x[2])
      word  = best[1].strip()
      score = float(best[2])

      print(f"✅ MOT RETENU : '{word}'  (confiance: {score:.2f})")
      print(f"   Unicode : {[hex(ord(c)) for c in word]}\n")

      return word, score