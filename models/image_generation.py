import requests
import io
from PIL import Image
import os
from dotenv import load_dotenv
load_dotenv()

HF_API_KEY = os.getenv('HF_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
# def generate_image(topic):
#     """Generate an image using Stable Diffusion via Hugging Face API."""
#     url = "https://api-inference.huggingface.co/stabilityai/stable-code-3b"
#     headers = {"Authorization": f"Bearer {HF_API_KEY}"}
#     payload = {"inputs": f"Illustration of {topic}, highly detailed."}

#     response = requests.post(url, headers=headers, json=payload)
#     if response.status_code == 200:
#         image = Image.open(io.BytesIO(response.content))
#         image_path = f"{topic.replace(' ', '_')}.png"
#         image.save(image_path)
#         return f"Image generated and saved as {image_path}."
#     return "Image generation failed."


# def generate_image(topic):
#     """Generate an image representation for the topic using the Groq API."""
#     url = "https://api.groq.com/v1/images/generate"
#     headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
#     payload = {
#         "model": "llama-3.2-90b-vision-preview",  # Specify the correct model name
#         "prompt": f"Illustration of {topic}, highly detailed."
#     }

    
#     response = requests.post(url, headers=headers, json=payload)
    
#     if response.status_code == 200:
#         image = Image.open(io.BytesIO(response.content))
#         image_path = f"{topic.replace(' ', '_')}.png"
#         image.save(image_path)
#         return f"Image generated and saved as {image_path}."
    
#     return "Image generation failed."

from diffusers import StableDiffusionPipeline
import torch

def generate_image(topic):
    """Generate an image representation for the topic using Stable Diffusion."""
    
    model_id = "runwayml/stable-diffusion-v1-5"  # Open-source model
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    prompt = f"Highly detailed illustration of {topic}, professional, concept art."
    image = pipe(prompt).images[0]

    image_path = f"{topic.replace(' ', '_')}.png"
    image.save(image_path)
    
    return f"Image generated and saved as {image_path}."

# import requests

# def generate_image(topic):
#     """Generate an image using Stability AI's Stable Diffusion API."""
    
#     url = "https://api.stability.ai/v2beta/stable-image/generate/core"
#     headers = {
#         "Authorization": f"Bearer YOUR_STABILITY_AI_KEY",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": "stable-diffusion-xl-beta-v2-2-2",
#         "prompt": f"Highly detailed illustration of {topic}, concept art, professional.",
#         "width": 1024,
#         "height": 1024,
#         "steps": 30
#     }

#     response = requests.post(url, headers=headers, json=payload)

#     if response.status_code == 200:
#         image_url = response.json()["image"]  # Assuming response gives a direct image URL
#         return f"Image generated: {image_url}"
    
#     return "Image generation failed."
