import os
import pandas as pd
from collections import defaultdict
import re

# Select path
folder = "/home/xrad/KI_Piktogramm/data/Tuebingen/Pat ohne Piktogramm"




patient_counts = defaultdict(int)


for filename in os.listdir(folder):
    
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        
        match = re.search(r'Pat\d+', filename)
        if match:
            patient_id = match.group()  # z. B. "Pat62"
            patient_counts[patient_id] += 1

df = pd.DataFrame(list(patient_counts.items()), columns=["Patient", "Number of Images"])

# Sort by patient number
df["Number"] = df["Patient"].apply(lambda x: int(re.search(r'\d+', x).group()))
df = df.sort_values(by="Number").drop(columns=["Number"])

# Save as Excel file
excel_path = os.path.join(folder, "Patienten_Tuebingen_ohne_Piktogramm.xlsx")
df.to_excel(excel_path, index=False)

print("Saved Excel file: Patienten_Tuebingen_ohne_Piktogramm.xlsx")