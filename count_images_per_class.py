import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from collections import Counter
import re

CLASS_NAMES = {
    1: "Oberbauchlängsschnitt links",
    2: "Oberbauchlängsschnitt MCL rechts",
    3: "Subcostaler Schrägschnitt rechts",
    4: "Intercostalschnitt rechts",
    5: "Hoher Flankenschnitt rechts",
}

root = tk.Tk()
root.withdraw()

folder = filedialog.askdirectory(
    title="Ordner mit den Ultraschallbildern auswählen"
)

if folder:
    counts = Counter()
    unmatched = 0

    for file in Path(folder).iterdir():
        if not file.is_file():
            continue

        match = re.search(r"_([1-5])$", file.stem)

        if match:
            counts[int(match.group(1))] += 1
        else:
            unmatched += 1

    print(f"\nOrdner: {folder}\n")
    total = 0

    for digit, name in CLASS_NAMES.items():
        count = counts[digit]
        total += count
        print(f"{name} ({digit}): {count} Dateien")

    print(f"\nGesamtzahl zugeordneter Dateien: {total}")
    print(f"Dateien ohne passende Endziffer: {unmatched}")

else:
    print("Kein Ordner ausgewählt.")