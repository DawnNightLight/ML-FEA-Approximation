import csv
from pathlib import Path

# Folder containing the files
folder = Path(r"C:\Users\murra\PycharmProjects\pythonProject1\physics\Misc\STEP Files\FinrayManyVariants")

# CSV file
csv_file = Path(r"C:\Users\murra\PycharmProjects\pythonProject1\physics\Code\CAD\invalidVariants.csv")

with csv_file.open(newline="", encoding="utf-8") as f:
    reader = csv.reader(f)

    for row in reader:
        # Skip empty rows
        if not row:
            continue

        # Build filename: 1-2-3-4.txt
        filename = "-".join(value.strip() for value in row) + ".step"
        file_path = folder / filename

        if file_path.exists():
            file_path.unlink()
            print(f"Deleted: {file_path}")
        else:
            print(f"Not found: {file_path}")