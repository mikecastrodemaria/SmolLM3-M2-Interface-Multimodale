"""
Gradio Interface for SmolLM3 (text) and SmolVLM2 (vision)
Compatible with Mac, Windows, and Linux with automatic CPU/GPU detection
Fixed version - ClearButton bug resolved
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import platform
import sys

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🖥️ Device detected: {DEVICE}")
print(f"💻 System: {platform.system()} {platform.machine()}")
print(f"🐍 Python: {sys.version.split()[0]}")

# Check versions
try:
    import transformers
    import gradio as gr_check
    print(f"📦 Transformers: {transformers.__version__}")
    print(f"📦 Gradio: {gr_check.__version__}")

    # Check minimum version
    trans_version = tuple(map(int, transformers.__version__.split('.')[:2]))
    grad_version = tuple(map(int, gr_check.__version__.split('.')[:2]))

    if trans_version < (4, 45):
        print("⚠️  WARNING: Transformers version too old!")
        print("   Run: pip install --upgrade transformers>=4.45.0")

    if grad_version < (4, 0):
        print("⚠️  WARNING: Gradio version too old!")
        print("   Run: pip install --upgrade gradio>=4.0.0")

except Exception as e:
    print(f"⚠️  Error during version check: {e}")

# Models to load
TEXT_MODEL = "HuggingFaceTB/SmolLM3-3B"  # Instruct version (no -Instruct in the name)
VISION_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

# Alternative: use quantized versions if loading issues
# TEXT_MODEL = "ggml-org/SmolLM3-3B-GGUF"  # Quantized version
# VISION_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

# Global variables for models
text_model = None
text_tokenizer = None
vision_model = None
vision_processor = None

def load_text_model():
    """Load SmolLM3 text model"""
    global text_model, text_tokenizer

    if text_model is None:
        print("📥 Loading SmolLM3-3B-Instruct...")
        try:
            text_tokenizer = AutoTokenizer.from_pretrained(
                TEXT_MODEL,
                trust_remote_code=True
            )

            # Optimized loading based on device
            if DEVICE == "cuda":
                try:
                    text_model = AutoModelForCausalLM.from_pretrained(
                        TEXT_MODEL,
                        dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                except:
                    # Fallback without device_map if accelerate is not available
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

            print("✅ SmolLM3 loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading SmolLM3: {e}")
            raise

    return text_model, text_tokenizer

def load_vision_model():
    """Load SmolVLM2 vision model"""
    global vision_model, vision_processor

    if vision_model is None:
        print("📥 Loading SmolVLM2-2.2B-Instruct...")
        try:
            # Load processor with trust_remote_code
            vision_processor = AutoProcessor.from_pretrained(
                VISION_MODEL,
                trust_remote_code=True
            )

            # Optimized loading based on device
            if DEVICE == "cuda":
                try:
                    vision_model = AutoModelForImageTextToText.from_pretrained(
                        VISION_MODEL,
                        dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                except:
                    # Fallback without device_map if accelerate is not available
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

            print("✅ SmolVLM2 loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading SmolVLM2: {e}")
            print("\n💡 Make sure you have the correct version:")
            print("   pip install --upgrade transformers>=4.45.0")
            raise

    return vision_model, vision_processor

def generate_text(prompt, max_length=512, temperature=0.7, top_p=0.9, think_mode="/think"):
    """Generate text with SmolLM3, returns (answer, thinking_trace)"""
    if not prompt or prompt.strip() == "":
        return "⚠️ Please enter a prompt.", ""

    if think_mode not in ["/think", "/no_think"]:
        think_mode = "/think"

    try:
        model, tokenizer = load_text_model()

        # Format prompt for instruct model
        messages = [
            {"role": "system", "content": think_mode},
            {"role": "user", "content": prompt}
        ]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenization
        inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)

        # Generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decoding
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response
        if "assistant" in generated_text.lower():
            parts = generated_text.lower().split("assistant")
            if len(parts) > 1:
                response = generated_text[generated_text.lower().rfind("assistant") + 9:].strip()
            else:
                response = generated_text
        else:
            response = generated_text.replace(input_text, "").strip()

        # If think mode is enabled, try to separate thinking from answer
        thinking_trace = ""
        final_answer = response

        if think_mode == "/think":
            # Try to find the thinking tags or patterns
            # SmolLM3 uses <think>...</think> tags for reasoning
            if "<think>" in response and "</think>" in response:
                # Extract thinking content (between the tags)
                think_start = response.find("<think>") + len("<think>")
                think_end = response.find("</think>")
                thinking_trace = response[think_start:think_end].strip()
                # Extract final answer (everything after </think>)
                final_answer = response[response.find("</think>") + len("</think>"):].strip()
            elif response:
                # If no tags found, treat entire response as answer
                final_answer = response
                thinking_trace = ""
        else:
            # If /no_think mode, remove any thinking tags if they appear
            if "<think>" in response and "</think>" in response:
                # Extract only the final answer (after </think>)
                final_answer = response[response.find("</think>") + len("</think>"):].strip()
                thinking_trace = ""
            else:
                final_answer = response
                thinking_trace = ""

        return final_answer if final_answer else response, thinking_trace

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Full error:\n{error_detail}")
        return f"❌ Error: {str(e)}\n\nCheck the console for more details.", ""

def analyze_image(image, question, max_length=256):
    """Analyze an image with SmolVLM2"""
    try:
        if image is None:
            return "⚠️ Please upload an image."

        if not question or question.strip() == "":
            return "⚠️ Please ask a question about the image."

        model, processor = load_vision_model()

        # Prepare image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        # Format prompt for SmolVLM2
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

        # Process image and text
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False
            )

        # Decoding
        generated_text = processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Extract response
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
        print(f"Full error:\n{error_detail}")

        # Detailed error message for user
        error_msg = f"❌ Error: {str(e)}\n\n"

        if "Unrecognized processing class" in str(e):
            error_msg += "💡 Solution: Update transformers:\n"
            error_msg += "   pip install --upgrade transformers>=4.45.0\n\n"

        error_msg += "Check the console for more details."
        return error_msg

# Interface Gradio
with gr.Blocks(title="SmolLM3 & SmolVLM2", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🤖 SmolLM3 & SmolVLM2 - Multimodal Interface

    Complete interface for text analysis and image processing with **SmolLM3** and **SmolVLM2** models from HuggingFace.

    **Features:**
    - 💬 Text Mode: Text generation with SmolLM3-3B (instruct version)
    - 🧠 Extended Thinking: See the model's reasoning process (enabled by default)
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
                    
                    # Add checkbox for think mode
                    think_mode_checkbox = gr.Checkbox(
                        value=True,
                        label="Enable Extended Thinking",
                        info="Show the model's reasoning process in an expandable section below the response."
                    )

                    text_submit = gr.Button("🚀 Generate", variant="primary")

                with gr.Column():
                    text_output = gr.Textbox(
                        label="Response",
                        lines=15,
                        interactive=False
                    )
                    # Add accordion for thinking trace
                    with gr.Accordion("🧠 Thinking Process", open=False, visible=True) as thinking_accordion:
                        thinking_output = gr.Textbox(
                            label="Reasoning Trace",
                            lines=10,
                            interactive=False,
                            placeholder="The model's reasoning process will appear here when using Extended Thinking mode..."
                        )

            # Clear button after component definition
            with gr.Row():
                clear_text_btn = gr.Button("🗑️ Clear All")
                clear_text_btn.click(
                    fn=lambda: ("", "", "", gr.Accordion(visible=False)),
                    inputs=[],
                    outputs=[text_input, text_output, thinking_output, thinking_accordion]
                )
            
            gr.Markdown("""
            **Prompt Examples:**
            - *"Write a short story about a robot discovering friendship"*
            - *"Explain how neural networks work"*
            - *"Code a Python function to calculate the Fibonacci sequence"*

            **💡 Tip:** With Extended Thinking mode enabled (default), the model shows its reasoning process
            in the "Thinking Process" accordion below the response. This helps you understand how it arrived at the answer!
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
            
            # Function to handle generation and accordion visibility
            def generate_and_update_visibility(prompt, max_length, temperature, think_enabled):
                answer, thinking = generate_text(
                    prompt, max_length, temperature, 0.9, "/think" if think_enabled else "/no_think"
                )
                # Show accordion only if thinking is enabled AND there's actual thinking content
                show_accordion = think_enabled and thinking.strip() != ""
                return answer, thinking, gr.Accordion(visible=show_accordion)

            text_submit.click(
                fn=generate_and_update_visibility,
                inputs=[text_input, text_max_length, text_temperature, think_mode_checkbox],
                outputs=[text_output, thinking_output, thinking_accordion]
            )

            # Also update accordion visibility when checkbox is toggled
            think_mode_checkbox.change(
                fn=lambda enabled: gr.Accordion(visible=enabled),
                inputs=[think_mode_checkbox],
                outputs=[thinking_accordion]
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
                        value="Describe this image and its style in a very detailed manner, follow the format of describing: what, who, where, when, how. You don't need to fill in all if they are irrelevant. Please remove What, Who, Where, When, How prefixes and make it one paragraph.",
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
