"""
Interface Gradio pour SmolLM3 (texte) et SmolVLM2 (vision)
Compatible Mac, Windows et Linux avec détection automatique CPU/GPU
Version corrigée - Bug ClearButton résolu
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import platform
import sys

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🖥️ Appareil détecté: {DEVICE}")
print(f"💻 Système: {platform.system()} {platform.machine()}")
print(f"🐍 Python: {sys.version.split()[0]}")

# Vérifier les versions
try:
    import transformers
    import gradio as gr_check
    print(f"📦 Transformers: {transformers.__version__}")
    print(f"📦 Gradio: {gr_check.__version__}")
    
    # Vérifier version minimale
    trans_version = tuple(map(int, transformers.__version__.split('.')[:2]))
    grad_version = tuple(map(int, gr_check.__version__.split('.')[:2]))
    
    if trans_version < (4, 45):
        print("⚠️  ATTENTION: Transformers version trop ancienne!")
        print("   Exécutez: pip install --upgrade transformers>=4.45.0")
    
    if grad_version < (4, 0):
        print("⚠️  ATTENTION: Gradio version trop ancienne!")
        print("   Exécutez: pip install --upgrade gradio>=4.0.0")
        
except Exception as e:
    print(f"⚠️  Erreur lors de la vérification: {e}")

# Modèles à charger
TEXT_MODEL = "HuggingFaceTB/SmolLM3-3B"  # Version instruct (pas -Instruct dans le nom)
VISION_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

# Alternative : utiliser les versions quantifiées si problème de chargement
# TEXT_MODEL = "ggml-org/SmolLM3-3B-GGUF"  # Version quantifiée
# VISION_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

# Variables globales pour les modèles
text_model = None
text_tokenizer = None
vision_model = None
vision_processor = None

def load_text_model():
    """Charge le modèle texte SmolLM3"""
    global text_model, text_tokenizer
    
    if text_model is None:
        print("📥 Chargement de SmolLM3-3B-Instruct...")
        try:
            text_tokenizer = AutoTokenizer.from_pretrained(
                TEXT_MODEL,
                trust_remote_code=True
            )
            
            # Chargement optimisé selon le device
            if DEVICE == "cuda":
                try:
                    text_model = AutoModelForCausalLM.from_pretrained(
                        TEXT_MODEL,
                        dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                except:
                    # Fallback sans device_map si accelerate n'est pas disponible
                    text_model = AutoModelForCausalLM.from_pretrained(
                        TEXT_MODEL,
                        dtype=torch.float16,
                        trust_remote_code=True
                    ).to(DEVICE)
            elif DEVICE == "mps":
                text_model = AutoModelForCausalLM.from_pretrained(
                    TEXT_MODEL,
                    dtype=torch.float16,
                    trust_remote_code=True
                ).to(DEVICE)
            else:
                text_model = AutoModelForCausalLM.from_pretrained(
                    TEXT_MODEL,
                    dtype=torch.float32,
                    trust_remote_code=True
                ).to(DEVICE)
            
            print("✅ SmolLM3 chargé avec succès!")
        except Exception as e:
            print(f"❌ Erreur lors du chargement de SmolLM3: {e}")
            raise
    
    return text_model, text_tokenizer

def load_vision_model():
    """Charge le modèle vision SmolVLM2"""
    global vision_model, vision_processor
    
    if vision_model is None:
        print("📥 Chargement de SmolVLM2-2.2B-Instruct...")
        try:
            # Charger le processeur avec trust_remote_code
            vision_processor = AutoProcessor.from_pretrained(
                VISION_MODEL,
                trust_remote_code=True
            )
            
            # Chargement optimisé selon le device
            if DEVICE == "cuda":
                try:
                    vision_model = AutoModelForImageTextToText.from_pretrained(
                        VISION_MODEL,
                        dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                except:
                    # Fallback sans device_map si accelerate n'est pas disponible
                    vision_model = AutoModelForImageTextToText.from_pretrained(
                        VISION_MODEL,
                        dtype=torch.float16,
                        trust_remote_code=True
                    ).to(DEVICE)
            elif DEVICE == "mps":
                vision_model = AutoModelForImageTextToText.from_pretrained(
                    VISION_MODEL,
                    dtype=torch.float16,
                    trust_remote_code=True
                ).to(DEVICE)
            else:
                vision_model = AutoModelForImageTextToText.from_pretrained(
                    VISION_MODEL,
                    dtype=torch.float32,
                    trust_remote_code=True
                ).to(DEVICE)
            
            print("✅ SmolVLM2 chargé avec succès!")
        except Exception as e:
            print(f"❌ Erreur lors du chargement de SmolVLM2: {e}")
            print("\n💡 Vérifiez que vous avez la bonne version:")
            print("   pip install --upgrade transformers>=4.45.0")
            raise
    
    return vision_model, vision_processor

def generate_text(prompt, max_length=512, temperature=0.7, top_p=0.9):
    """Génère du texte avec SmolLM3"""
    if not prompt or prompt.strip() == "":
        return "⚠️ Veuillez entrer un prompt."
    
    try:
        model, tokenizer = load_text_model()
        
        # Format du prompt pour le modèle instruct
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenization
        inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
        
        # Génération
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Décodage
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire seulement la réponse de l'assistant
        if "assistant" in generated_text.lower():
            parts = generated_text.lower().split("assistant")
            if len(parts) > 1:
                response = generated_text[generated_text.lower().rfind("assistant") + 9:].strip()
            else:
                response = generated_text
        else:
            response = generated_text.replace(input_text, "").strip()
        
        return response if response else generated_text
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Erreur complète:\n{error_detail}")
        return f"❌ Erreur: {str(e)}\n\nVérifiez la console pour plus de détails."

def analyze_image(image, question, max_length=256):
    """Analyse une image avec SmolVLM2"""
    try:
        if image is None:
            return "⚠️ Veuillez télécharger une image."
        
        if not question or question.strip() == "":
            return "⚠️ Veuillez poser une question sur l'image."
        
        model, processor = load_vision_model()
        
        # Préparer l'image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")
        
        # Format du prompt pour SmolVLM2
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Traitement de l'image et du texte
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Génération
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False
            )
        
        # Décodage
        generated_text = processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Extraire la réponse
        if "Assistant:" in generated_text:
            response = generated_text.split("Assistant:")[-1].strip()
        elif question in generated_text:
            response = generated_text.split(question)[-1].strip()
        else:
            response = generated_text
        
        return response if response else generated_text
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Erreur complète:\n{error_detail}")
        
        # Message d'erreur détaillé pour l'utilisateur
        error_msg = f"❌ Erreur: {str(e)}\n\n"
        
        if "Unrecognized processing class" in str(e):
            error_msg += "💡 Solution: Mettez à jour transformers:\n"
            error_msg += "   pip install --upgrade transformers>=4.45.0\n\n"
        
        error_msg += "Consultez la console pour plus de détails."
        return error_msg

# Interface Gradio
with gr.Blocks(title="SmolLM3 & SmolVLM2", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🤖 SmolLM3 & SmolVLM2 - Multimodal Interface

    Complete interface for text analysis and image processing with **SmolLM3** and **SmolVLM2** models from HuggingFace.

    **Features:**
    - 💬 Text Mode: Text generation with SmolLM3-3B (instruct version)
    - 👁️ Vision Mode: Image analysis with SmolVLM2-2.2B-Instruct
    - ⚡ Compatible with CPU, CUDA GPU, and Apple Silicon (MPS)
    - 🌍 Multilingual support (EN, FR, ES, DE, IT, PT)
    """)
    
    with gr.Tabs():
        # Text Mode Tab
        with gr.Tab("💬 Text Mode"):
            gr.Markdown("### Text generation with SmolLM3")
            
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Your question or prompt",
                        placeholder="Example: Explain relativity in simple terms...",
                        lines=5
                    )
                    
                    with gr.Row():
                        text_max_length = gr.Slider(
                            minimum=50,
                            maximum=1024,
                            value=512,
                            step=50,
                            label="Max Length"
                        )
                        text_temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature (creativity)"
                        )
                    
                    text_submit = gr.Button("🚀 Generate", variant="primary")

                with gr.Column():
                    text_output = gr.Textbox(
                        label="Response",
                        lines=15,
                        interactive=False
                    )
            
            # Clear button after component definition
            with gr.Row():
                clear_text_btn = gr.Button("🗑️ Clear All")
                clear_text_btn.click(
                    fn=lambda: ("", ""),
                    inputs=[],
                    outputs=[text_input, text_output]
                )
            
            gr.Markdown("""
            **Prompt Examples:**
            - *"Write a short story about a robot discovering friendship"*
            - *"Explain how neural networks work"*
            - *"Code a Python function to calculate the Fibonacci sequence"*
            """)
            
            # Predefined examples
            gr.Examples(
                examples=[
                    ["Explain general relativity in simple terms", 300, 0.7],
                    ["Write a haiku about artificial intelligence", 100, 0.9],
                    ["What is the difference between Python and JavaScript?", 400, 0.5],
                ],
                inputs=[text_input, text_max_length, text_temperature]
            )
            
            text_submit.click(
                fn=generate_text,
                inputs=[text_input, text_max_length, text_temperature],
                outputs=text_output
            )
        
        # Vision Mode Tab
        with gr.Tab("👁️ Vision Mode"):
            gr.Markdown("### Image analysis with SmolVLM2")
            
            with gr.Row():
                with gr.Column():
                    vision_image = gr.Image(
                        type="pil",
                        label="Upload an image"
                    )
                    vision_question = gr.Textbox(
                        label="Your question about the image",
                        placeholder="Example: Describe this image in detail...",
                        lines=3
                    )
                    vision_max_length = gr.Slider(
                        minimum=50,
                        maximum=512,
                        value=256,
                        step=50,
                        label="Max response length"
                    )
                    vision_submit = gr.Button("🔍 Analyze", variant="primary")
                
                with gr.Column():
                    vision_output = gr.Textbox(
                        label="Analysis",
                        lines=15,
                        interactive=False
                    )

            # Clear button after component definition
            with gr.Row():
                clear_vision_btn = gr.Button("🗑️ Clear All")
                clear_vision_btn.click(
                    fn=lambda: (None, "", ""),
                    inputs=[],
                    outputs=[vision_image, vision_question, vision_output]
                )
            
            gr.Markdown("""
            **SmolVLM2 Capabilities:**
            - Detailed image descriptions
            - Visual question answering
            - OCR and text reading in images
            - Object counting
            - Document and chart analysis
            """)
            
            vision_submit.click(
                fn=analyze_image,
                inputs=[vision_image, vision_question, vision_max_length],
                outputs=vision_output
            )
    
    gr.Markdown(f"""
    ---
    **System Information:**
    - Device: `{DEVICE}`
    - Platform: `{platform.system()} {platform.machine()}`
    - Python: `{sys.version.split()[0]}`
    - Text Model: `{TEXT_MODEL}`
    - Vision Model: `{VISION_MODEL}`

    💡 **Note:** Models load automatically on first use of each mode.

    ⚠️ **In case of error:** Make sure you have installed `transformers>=4.53.0` and `gradio>=4.0.0`
    """)

if __name__ == "__main__":
    print("\n🚀 Launching Gradio interface...")
    print(f"📍 Interface will be available at: http://127.0.0.1:7860")
    print("\n" + "="*60)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )