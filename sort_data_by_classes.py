import os
import shutil
import re

# Select input path and output path
source_folder = r"/home/xrad/KI_Piktogramm/data/Tuebingen/Training/Patient_mit_Piktogramm"
output_base = r"/home/xrad/KI_Piktogramm/data/Tuebingen/Training/data_sorted_by_classes"

# Mapping: create folder names
folder_names = {
    1: "Oberbauchlaengsschnitt_links",
    2: "Oberbauchlaengsschnitt_MCL_rechts",
    3: "Subcostaler Schraegschnitt_rechts",
    4: "Intercostalschnitt_rechts",
    5: "Hoher_Flankenschnitt_rechts"
}

# Create output folder
for name in folder_names.values():
    os.makedirs(os.path.join(output_base, name), exist_ok=True)

# Sort data
for filename in os.listdir(source_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        match = re.search(r"_(\d+)\.(jpg|jpeg|png)$", filename, re.IGNORECASE)
        if match:
            last_number = int(match.group(1))
            if last_number in folder_names:
                src = os.path.join(source_folder, filename)
                dst = os.path.join(output_base, folder_names[last_number], filename)
                shutil.move(src, dst) 
                print(f"{filename} → {folder_names[last_number]}")