import torch
from mingpt.model import GPT
from mingpt.utils import sample

# Load the fine-tuned model
model = GPT()
model.load_state_dict(torch.load('./fine_tuned_model.pth'))
model.eval()  # Set the model to evaluation mode

def recommend_roles(skills):
    query = f"What role matches the skills {skills}?"
    tokens = model.tokenizer.encode(query)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    # Generate predictions
    output = sample(model, tokens, steps=50)
    recommendation = model.tokenizer.decode(output[0].tolist())
    return recommendation.strip()

# Example usage
skills = "Python, Machine Learning"
print("Recommended Role:", recommend_roles(skills))
