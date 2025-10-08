# 📦 Guide d'Installation - SmolLM3 & SmolVLM2

Guide complet pour installer et exécuter l'interface Gradio sur **macOS** et **Windows 11**.

---

## 🍎 Installation sur macOS

### Prérequis
- macOS 11.0 (Big Sur) ou supérieur
- Pour Apple Silicon (M1/M2/M3): utilisation automatique de MPS
- Pour Intel Mac: utilisation CPU ou GPU externe

### Étape 1: Installer Homebrew (si pas déjà installé)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Étape 2: Installer Python 3.10+

```bash
# Installer Python via Homebrew
brew install python@3.11

# Vérifier l'installation
python3 --version
```

### Étape 3: Créer un environnement virtuel

```bash
# Créer un dossier pour le projet
mkdir ~/smollm-app
cd ~/smollm-app

# Créer un environnement virtuel
python3 -m venv venv

# Activer l'environnement
source venv/bin/activate
```

### Étape 4: Installer les dépendances

```bash
# Mettre à jour pip
pip install --upgrade pip

# Installer PyTorch pour Mac (avec support MPS pour Apple Silicon)
pip install torch torchvision torchaudio

# Installer les autres dépendances
pip install transformers>=4.53.0 gradio>=4.0.0 accelerate pillow sentencepiece protobuf

# Pour optimiser les performances sur Apple Silicon
pip install accelerate bitsandbytes
```

### Étape 5: Télécharger et lancer l'application

```bash
# Créer le fichier Python (copier le code de l'artefact)
nano app.py
# Coller le code, puis Ctrl+O pour sauvegarder, Ctrl+X pour quitter

# Ou télécharger directement si vous avez le fichier
# curl -O https://votre-url/app.py

# Lancer l'application
python app.py
```

### Étape 6: Accéder à l'interface

Ouvrez votre navigateur et allez sur: **http://localhost:7860**

---

## 🪟 Installation sur Windows 11

### Prérequis
- Windows 11 (ou Windows 10)
- Pour GPU NVIDIA: Pilotes NVIDIA récents + CUDA Toolkit
- Pour CPU uniquement: aucune configuration GPU nécessaire

### Étape 1: Installer Python 3.10+

1. Téléchargez Python depuis [python.org](https://www.python.org/downloads/)
2. Lors de l'installation, **cochez "Add Python to PATH"**
3. Vérifiez l'installation:

```cmd
python --version
```

### Étape 2: Créer un environnement virtuel

```cmd
# Créer un dossier pour le projet
mkdir C:\smollm-app
cd C:\smollm-app

# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
venv\Scripts\activate
```

### Étape 3: Installer les dépendances

#### Option A: Avec GPU NVIDIA (recommandé si disponible)

```cmd
# Mettre à jour pip
python -m pip install --upgrade pip

# Installer PyTorch avec support CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Installer les autres dépendances
pip install transformers>=4.53.0 gradio>=4.0.0 accelerate pillow sentencepiece protobuf

# Optionnel: pour quantification et optimisation
pip install bitsandbytes-windows
```

#### Option B: CPU uniquement (si pas de GPU)

```cmd
# Mettre à jour pip
python -m pip install --upgrade pip

# Installer PyTorch version CPU
pip install torch torchvision torchaudio

# Installer les autres dépendances
pip install transformers>=4.53.0 gradio>=4.0.0 accelerate pillow sentencepiece protobuf
```

### Étape 4: Télécharger et lancer l'application

```cmd
# Créer le fichier Python
notepad app.py
# Coller le code de l'artefact et sauvegarder

# Ou utiliser PowerShell pour télécharger
# Invoke-WebRequest -Uri "https://votre-url/app.py" -OutFile "app.py"

# Lancer l'application
python app.py
```

### Étape 5: Accéder à l'interface

Ouvrez votre navigateur et allez sur: **http://localhost:7860**

---

## ⚙️ Configuration avancée

### Réduire l'utilisation de la mémoire

Si vous avez des problèmes de RAM/VRAM, modifiez ces lignes dans `app.py`:

```python
# Utiliser une quantification 8-bit (nécessite bitsandbytes)
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

# Puis lors du chargement:
text_model = AutoModelForCausalLM.from_pretrained(
    TEXT_MODEL,
    quantization_config=quantization_config,
    device_map="auto"
)
```

### Accéder depuis d'autres appareils sur le réseau

Modifiez la dernière ligne de `app.py`:

```python
demo.launch(
    server_name="0.0.0.0",  # Écouter sur toutes les interfaces
    server_port=7860,
    share=True  # Créer un lien public temporaire (via Gradio)
)
```

### Variables d'environnement utiles

```bash
# Pour Mac/Linux
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Fallback CPU si MPS échoue
export TRANSFORMERS_CACHE=~/smollm-cache  # Cache des modèles

# Pour Windows (cmd)
set TRANSFORMERS_CACHE=C:\smollm-cache

# Pour Windows (PowerShell)
$env:TRANSFORMERS_CACHE="C:\smollm-cache"
```

---

## 🚀 Utilisation

### Mode Texte
1. Allez dans l'onglet "💬 Mode Texte"
2. Tapez votre question ou prompt
3. Ajustez les paramètres (longueur, température)
4. Cliquez sur "🚀 Générer"

### Mode Vision
1. Allez dans l'onglet "👁️ Mode Vision"
2. Téléchargez une image
3. Posez une question sur l'image
4. Cliquez sur "🔍 Analyser"

---

## ❓ Dépannage

### Problème: "No module named 'transformers'"
**Solution:** Assurez-vous que l'environnement virtuel est activé et réinstallez:
```bash
pip install transformers>=4.53.0
```

### Problème: "CUDA out of memory"
**Solution:** 
- Réduire la longueur maximale de génération
- Utiliser la quantification 8-bit
- Fermer les autres applications gourmandes en mémoire

### Problème: Vitesse lente sur Mac M1/M2/M3
**Solution:**
```bash
# Vérifier que MPS est utilisé
python -c "import torch; print(f'MPS disponible: {torch.backends.mps.is_available()}')"

# Si False, réinstaller PyTorch:
pip install --upgrade torch torchvision torchaudio
```

### Problème: ModuleNotFoundError pour 'sentencepiece'
**Solution:**
```bash
pip install sentencepiece protobuf
```

### Problème: L'interface ne se charge pas
**Solution:**
1. Vérifiez les logs dans le terminal
2. Essayez un autre port:
```python
demo.launch(server_port=8080)
```
3. Désactivez le pare-feu temporairement

---

## 📊 Configuration système recommandée

### Minimum (CPU uniquement)
- **RAM:** 16 GB
- **Stockage:** 20 GB libres
- **Temps de génération:** ~30-60 secondes

### Recommandé (avec GPU)
- **RAM:** 16 GB
- **VRAM:** 8 GB (NVIDIA RTX 3060 ou supérieur)
- **Stockage:** 20 GB libres
- **Temps de génération:** ~2-5 secondes

### Optimal (Apple Silicon)
- **Modèle:** M1 Pro/Max, M2, M3 ou supérieur
- **RAM unifiée:** 16 GB minimum (32 GB recommandé)
- **Stockage:** 20 GB libres
- **Temps de génération:** ~3-8 secondes

---

## 📚 Ressources additionnelles

- **Documentation SmolLM3:** https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Instruct
- **Documentation SmolVLM2:** https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
- **Gradio Docs:** https://www.gradio.app/docs
- **Transformers Docs:** https://huggingface.co/docs/transformers

---

## 🔄 Mise à jour

Pour mettre à jour l'application:

```bash
# Activer l'environnement
source venv/bin/activate  # Mac/Linux
# ou
venv\Scripts\activate  # Windows

# Mettre à jour les packages
pip install --upgrade transformers gradio torch

# Relancer l'application
python app.py
```

---

## 🎉 Bon usage !

L'interface devrait maintenant être opérationnelle. Les modèles seront téléchargés automatiquement au premier lancement (environ 10-15 GB au total).

**Temps de téléchargement estimé:**
- Connexion rapide (100 Mbps): ~15-20 minutes
- Connexion moyenne (50 Mbps): ~30-40 minutes
- Connexion lente (10 Mbps): 2-3 heures