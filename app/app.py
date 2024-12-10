# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:24:32 2024

@author: Aruna
"""

from flask import Flask, request, jsonify
import torch
from mingpt.model import GPT
from mingpt.bpe import get_encoder

app = Flask(__name__)

#career path guidance extension
# import json

# with open("career_paths.json", "r") as f:
#     career_path_data = json.load(f)

# Load the trained model
def load_trained_model(model_path, device):
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt2'
    model_config.vocab_size = 50257
    model_config.block_size = 1024
    model = GPT(model_config)
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights
    model.to(device)
    model.eval()
    return model

# Generate job role based on input skills
def generate_job_role(input_skills, model, bpe_encoder, device, max_length=100):
    tokenized_input = bpe_encoder.encode(input_skills)
    input_tensor = torch.tensor([tokenized_input], dtype=torch.long).to(device)
    with torch.no_grad():
        output_tensor = model.generate(input_tensor, max_length=max_length)
    output_tokens = output_tensor[0].tolist()
    return bpe_encoder.decode(output_tokens)

# Initialize model and encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bpe_encoder = get_encoder()
model_path = "path/to/your/trained_model.pth"
model = load_trained_model(model_path, device)

#added extensions
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     input_skills = data.get("skills", "")
#     if not input_skills:
#         return jsonify({"error": "Skills input is required"}), 400

#     # Generate job role suggestions
#     job_roles = generate_job_roles(input_skills, model, bpe_encoder, device)

#     # Add career path guidance
#     recommendations = []
#     for job_role in job_roles:
#         guidance = career_path_data.get(job_role, {})
#         recommendations.append({
#             "job_role": job_role,
#             "description": guidance.get("description", "No description available."),
#             "next_steps": guidance.get("next_steps", []),
#             "certifications": guidance.get("certifications", [])
#         })
    
#     return jsonify({
#         "skills": input_skills,
#         "recommendations": recommendations
#     })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_skills = data.get("skills", "")
    if not input_skills:
        return jsonify({"error": "Skills input is required"}), 400
    
    job_role = generate_job_role(input_skills, model, bpe_encoder, device)
    return jsonify({"skills": input_skills, "job_role": job_role})

if __name__ == "__main__":
    app.run(debug=True)
