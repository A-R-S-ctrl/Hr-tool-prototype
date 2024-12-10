from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch

# Define a custom Dataset class to handle your training data
class SkillsRoleDataset(Dataset):
    def __init__(self, filepath, tokenizer, block_size=1024):
        self.data = open(filepath, 'r').readlines()
        self.tokenizer = tokenizer  # Hugging Face GPT-2 tokenizer
        self.block_size = block_size
        print(f"Loaded {len(self.data)} examples from {filepath}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx].strip()
        input_text, output_text = text.split(" --> ")

        # Tokenize both input and output (skills and job role)
        tokenized_input = self.tokenizer.encode(input_text, truncation=True, padding='max_length', max_length=self.block_size)
        tokenized_output = self.tokenizer.encode(output_text, truncation=True, padding='max_length', max_length=self.block_size)

        return torch.tensor(tokenized_input), torch.tensor(tokenized_output)

def main():
    # Load pre-trained GPT-2 model and tokenizer from Hugging Face
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Prepare the dataset
    train_dataset = SkillsRoleDataset('training_data.txt', tokenizer)

    # Create DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Check the first batch of data
    for batch_idx, (input_tensor, output_tensor) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Input Tensor Shape: {input_tensor.shape}, Output Tensor Shape: {output_tensor.shape}")
        if batch_idx == 1:
            break

    # Set up the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Start fine-tuning (You can modify the training loop as needed)
    model.train()  # Set the model to training mode

    # Configure the optimizer (Adam for fine-tuning)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    for epoch in range(3):  # Set number of epochs
        for batch_idx, (input_tensor, output_tensor) in enumerate(train_loader):
            input_tensor, output_tensor = input_tensor.to(device), output_tensor.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_tensor, labels=output_tensor)
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item()}")

    print("Fine-tuning complete!")

if __name__ == '__main__':
    main()
