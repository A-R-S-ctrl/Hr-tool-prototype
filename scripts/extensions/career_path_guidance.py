# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 08:41:27 2024

@author: Aruna
"""

import pandas as pd

# Define a function to generate career paths
def generate_career_path(current_role, skills):
    next_roles = recommend_roles(skills)  # Use the fine-tuned model
    return {"current_role": current_role, "skills": skills, "next_roles": next_roles}

# Example input
roles = [
    {"current_role": "Junior Developer", "skills": "Python"},
    {"current_role": "Data Analyst", "skills": "Data Visualization"}
]

# Generate career path recommendations
career_paths = [generate_career_path(r['current_role'], r['skills']) for r in roles]

# Save to CSV
df = pd.DataFrame(career_paths)
df.to_csv('career_path_recommendations.csv', index=False)
print("Career path recommendations saved to career_path_recommendations.csv")
