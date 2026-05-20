# services/cnn_encoder.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import numpy as np
import os

class ArabicLetterEncoder(nn.Module):
    """EXACTEMENT le même modèle que dans Colab"""
    def __init__(self, embedding_dim=128):
        super(ArabicLetterEncoder, self).__init__()
        
        # Charger MobileNetV2 pré-entraîné
        self.backbone = models.mobilenet_v2(pretrained=True)
        
        # Modifier la première couche pour grayscale (1 canal)
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1,  # input channels (grayscale)
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Copier les poids (moyenne sur les 3 canaux RGB)
        with torch.no_grad():
            self.backbone.features[0][0].weight[:, 0, :, :] = (
                original_conv.weight.mean(dim=1)
            )
        
        # Récupérer la dimension des features
        num_features = self.backbone.classifier[1].in_features
        
        # Remplacer le classifier par notre tête de projection
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        return nn.functional.normalize(x, p=2, dim=1)


class CNNEncoderService:
    def __init__(self, model_path='data/arabic_encoder_calliar_FINETUNED.pth', embedding_dim=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ArabicLetterEncoder(embedding_dim=embedding_dim)
        
        # CHARGER LES POIDS FINE-TUNÉS
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ CNN fine-tuné chargé depuis {model_path}")
        else:
            print(f"❌ Modèle non trouvé: {model_path}")
            print("   Le modèle DOIT être fine-tuné sur Calliar!")
            raise FileNotFoundError(f"Modèle {model_path} non trouvé")
        
        self.model.to(self.device)
        self.model.eval()
        
        # MÊME transformation que dans Colab
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def encode_image(self, image_data):
        """Génère un embedding identique à ceux de Colab"""
        # Décoder l'image base64
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Transformer
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Générer l'embedding
        with torch.no_grad():
            embedding = self.model(tensor).cpu().numpy().flatten()
        
        return embedding