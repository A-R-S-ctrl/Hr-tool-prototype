import torch
from mingpt.model import GPT
from mingpt.bpe import get_encoder

# Load the trained model
def load_trained_model(model_path, device):
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt2'
    model_config.vocab_size = 50257
    model_config.block_size = 1024
    model = GPT(model_config)
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Generate job role based on input skills
def generate_job_role(input_skills, model, bpe_encoder, device, max_length=100):
    # Tokenize the input skills
    tokenized_input = bpe_encoder.encode(input_skills)
    input_tensor = torch.tensor([tokenized_input], dtype=torch.long).to(device)
    
    # Generate tokens
    with torch.no_grad():
        output_tensor = model.generate(input_tensor, max_length=max_length)
    
    # Decode the generated tokens
    output_tokens = output_tensor[0].tolist()
    output_text = bpe_encoder.decode(output_tokens)
    return output_text

#for extensions-recommendations and career path
# def generate_job_roles(input_skills, model, bpe_encoder, device, max_length=100, top_k=3):
#     tokenized_input = bpe_encoder.encode(input_skills)
#     input_tensor = torch.tensor([tokenized_input], dtype=torch.long).to(device)
    
#     job_roles = []
#     for _ in range(top_k):  # Generate multiple suggestions
#         with torch.no_grad():
#             output_tensor = model.generate(input_tensor, max_length=max_length)
#         output_tokens = output_tensor[0].tolist()
#         output_text = bpe_encoder.decode(output_tokens)
#         job_roles.append(output_text.strip())
#     return job_roles

def main():
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the BPE encoder
    bpe_encoder = get_encoder()
    
    # Load the trained model
    model_path = "path/to/your/trained_model.pth"  # Path to the trained model weights
    model = load_trained_model(model_path, device)
    
    # User input skills
    input_skills = "Python, Machine Learning, Data Analysis"  # Example input
    print(f"Input Skills: {input_skills}")
    
    # Generate job role
    job_role = generate_job_role(input_skills, model, bpe_encoder, device)
    print(f"Predicted Job Role: {job_role}")

if __name__ == "__main__":
    main()
