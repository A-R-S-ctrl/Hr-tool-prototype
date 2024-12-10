from mingpt.model import GPT
from mingpt.trainer import Trainer
import torch
from torch.utils.data import Dataset, DataLoader
from mingpt.bpe import get_encoder  # Import the BPE encoder
import numpy as np

# Define a custom Dataset class to handle your training data
class SkillsRoleDataset(Dataset):
    def __init__(self, filepath, bpe_encoder, block_size=1024):
        # Read all lines from the dataset
        self.data = open(filepath, 'r').readlines()
        self.bpe_encoder = bpe_encoder  # BPE encoder for tokenizing text
        self.block_size = block_size  # Maximum sequence length
        print(f"Loaded {len(self.data)} examples from {filepath}")  # Debugging line
    
    def __len__(self):
        # Return the total number of examples
        return len(self.data)
    
    def __getitem__(self, idx):
        # Retrieve a specific example by index
        text = self.data[idx].strip()

        # Split the line into input (skills) and output (job role)
        input_text, output_text = text.split(" --> ")

        # Tokenize the input (skills) and output (job role) using BPE encoder
        tokenized_input = self.bpe_encoder.encode(input_text)
        tokenized_output = self.bpe_encoder.encode(output_text)

        # Padding: ensure the sequence length is consistent
        padding_length_input = self.block_size - len(tokenized_input)
        padding_length_output = self.block_size - len(tokenized_output)

        if padding_length_input > 0:
            tokenized_input += [0] * padding_length_input
        else:
            tokenized_input = tokenized_input[:self.block_size]

        if padding_length_output > 0:
            tokenized_output += [0] * padding_length_output
        else:
            tokenized_output = tokenized_output[:self.block_size]

        # Debugging: Print tokenized data (for the first few samples)
        if idx < 5:  # Print first 5 samples
            print(f"Example {idx} -> Input: {input_text} -> Tokenized: {tokenized_input}")
            print(f"Output: {output_text} -> Tokenized: {tokenized_output}")

        # Return the tokenized input and output as tensors
        return torch.tensor(tokenized_input), torch.tensor(tokenized_output)

def main():
    # Initialize the BPE encoder (using the default vocabulary from GPT-2)
    bpe_encoder = get_encoder()  # This loads the BPE encoder
    
    # Prepare dataset: Loading the training data from 'training_data.txt'
    train_dataset = SkillsRoleDataset('training_data.txt', bpe_encoder)

    # Create a DataLoader to iterate through the dataset in batches
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Print the first batch of data to verify DataLoader works
    for batch_idx, (input_tensor, output_tensor) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Input Tensor Shape: {input_tensor.shape}, Output Tensor Shape: {output_tensor.shape}")
        if batch_idx == 1:  # Only print the first two batches
            break

    # Initialize the GPT model
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt2'  # Using GPT-2 model
    model_config.vocab_size = 50257  # GPT-2 vocab size
    model_config.block_size = 1024  # Input context length (max token size for the model)
    
    # Set the device to CUDA if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GPT(model_config)
    model.to(device)  # Move model to the appropriate device

    # Configure the Trainer with learning rate, iterations, and batch size
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4
    train_config.max_iters = 1000  # Number of training iterations
    train_config.batch_size = 32  # Batch size for training
    trainer = Trainer(train_config, model, train_dataset)

    # Print model parameters to confirm everything is set up
    print(f"Training model with {sum(p.numel() for p in model.parameters())} parameters")

    # Start training the model
    print("Starting training...")
    for i in range(train_config.max_iters):
        trainer.run()  # Run one step of training
        if i % 100 == 0:  # Print progress every 100 iterations
            print(f"Iteration {i}/{train_config.max_iters}")

    # Final message after training is complete
    print("Training complete!")

# Add the 'if __name__ == "__main__":' guard to ensure proper execution on Windows and other platforms
if __name__ == '__main__':
    main()
