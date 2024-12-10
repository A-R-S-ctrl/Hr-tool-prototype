# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:00:38 2024

@author: Aruna
"""

import pandas as pd

def prepare_dataset(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Display the first few rows to understand its structure
    print("Dataset Loaded:")
    print(data.head())
    
    # Combine the related skills columns into a single 'skills' column
    related_columns = [f'related_{i}' for i in range(1, 11)]  # related_1 to related_10
    data['skills'] = data[related_columns].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    
    # Create the training text: "skills --> job_role"
    data['training_text'] = data['skills'] + " --> " + data['name']
    
    # Save the training data to a text file
    data['training_text'].to_csv('training_data.txt', index=False, header=False)

    print("Dataset prepared and saved as 'training_data.txt'")

# Example usage
prepare_dataset('related_job_skills.csv')
