# Changelog - SmolLM3 & SmolVLM2 Interface

Toutes les modifications importantes apportées à ce projet seront documentées dans ce fichier.

## [Version 1.1.0] - 2025-01-08

### 🐛 Corrections de bugs critiques

#### SmolVLM2 Model Loading Fix
- **Problème résolu**: `ValueError: Unrecognized configuration class SmolVLMConfig for AutoModelForCausalLM`
- **Solution**: Changement de `AutoModelForCausalLM` vers `AutoModelForImageTextToText` pour SmolVLM2
- **Fichiers modifiés**: `smollm3-gradio-app.py` (lignes 9, 120, 128, 134, 140)
- **Impact**: SmolVLM2 se charge maintenant correctement et peut analyser des images

#### Deprecated Parameter Fix
- **Problème résolu**: Warning `torch_dtype is deprecated! Use dtype instead!`
- **Solution**: Remplacement de tous les `torch_dtype=` par `dtype=` dans le code
- **Fichiers modifiés**: `smollm3-gradio-app.py` (lignes 73, 81, 87, 93, 122, 130, 136, 142)
- **Impact**: Plus de warnings au chargement des modèles

#### Missing generate() Method Fix
- **Problème résolu**: `AttributeError: 'SmolVLMModel' object has no attribute 'generate'`
- **Solution**: Utilisation de la bonne classe AutoModel pour SmolVLM2
- **Impact**: La génération de texte depuis les images fonctionne maintenant

### 📦 Mises à jour des dépendances

- **transformers**: Version minimale mise à jour de `4.45.0` → `4.53.0`
  - Raison: Support complet de SmolVLM2 avec `AutoModelForImageTextToText`
- Ajout de `einops` dans requirements.txt
- Clarification sur les dépendances optionnelles (`accelerate`, `hf_xet`)

### 📖 Documentation

#### README.md
- Ajout d'une section "Recent Updates" en haut du fichier
- Amélioration de la section "Troubleshooting" avec les 3 nouvelles erreurs résolues
- Mise à jour des instructions d'installation manuelle
- Ajout de notes sur les versions minimales requises

#### installation-guide.md
- Ajout d'une note importante sur les changements (Janvier 2025)
- Mise à jour de toutes les commandes pip avec les bonnes versions
- Ajout d'une nouvelle section "Dépannage" avec les erreurs SmolVLM2
- Correction du nom du fichier principal: `app.py` → `smollm3-gradio-app.py`

#### Nouveau fichier: CHANGELOG.md
- Documentation complète de tous les changements
- Format standardisé pour suivre l'évolution du projet

### 🔧 Changements techniques détaillés

#### Imports mis à jour
```python
# Avant
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

# Après
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
```

#### Chargement de SmolVLM2 mis à jour
```python
# Avant
vision_model = AutoModelForCausalLM.from_pretrained(
    VISION_MODEL,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Après
vision_model = AutoModelForImageTextToText.from_pretrained(
    VISION_MODEL,
    dtype=torch.float16,
    trust_remote_code=True
)
```

### ⚠️ Breaking Changes

Aucun breaking change pour les utilisateurs finaux. Les changements sont uniquement au niveau du code source.

### 🔄 Migration Guide

Si vous utilisez une ancienne version:

1. **Mettez à jour transformers**:
   ```bash
   pip install --upgrade transformers>=4.53.0
   ```

2. **Téléchargez le nouveau code** depuis le repository

3. **Relancez l'application**:
   ```bash
   python smollm3-gradio-app.py
   ```

Aucune autre action nécessaire, les modèles en cache restent valides.

---

## [Version 1.0.0] - 2025-01-01

### 🎉 Version initiale

- Interface Gradio pour SmolLM3 (génération de texte)
- Interface Gradio pour SmolVLM2 (analyse d'images)
- Support multi-plateforme (Windows, macOS, Linux)
- Détection automatique du matériel (CUDA, MPS, CPU)
- Support multilingue (EN, FR, ES, DE, IT, PT)
- Scripts d'installation automatiques
- Documentation complète

---

## Légende

- 🐛 Correction de bugs
- ✨ Nouvelles fonctionnalités
- 📦 Mises à jour de dépendances
- 📖 Documentation
- 🔧 Changements techniques
- ⚠️ Breaking changes
- 🔄 Guide de migration
- 🎉 Versions majeures
