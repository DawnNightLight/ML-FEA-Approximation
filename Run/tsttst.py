import re
import math
import csv
import shutil
import subprocess
from pathlib import Path

FRD = Path("model_5.frd")

def extract_max_displacement(frd_file):
    with open(frd_file, 'r') as f:
        lines = f.readlines()

    # Step 1: Collect all 1PSTEP lines (the timeline/load increment markers)
    pstep_indices = [idx for idx, line in enumerate(lines) if "1PSTEP" in line]

    if not pstep_indices:
        print("Error: No 1PSTEP markers found in the file. Check if simulation executed.")
        return None, None

    target_disp_idx = -1
    print(pstep_indices)
    for i in pstep_indices:
        if "DISP" in lines[i+2]:
            print(i)
            disp_header_idx = i
            break
        else:
            return None, None

    max_magnitude = -1.0
    max_node_id = -1

    # REGEX PATTERN: Matches an optional minus, digits/dots, 'E', optional sign, digits.
    # This perfectly isolates scientific notation numbers like -2.57213E-03
    scientific_number_pattern = re.compile(r'-?\d+\.\d+[Ee][+-]\d+')

    # 2. Parse the node data lines inside this DISP block
    for idx in range(disp_header_idx + 1, len(lines)):
        line = lines[idx].strip()

        if line.startswith("-3") or "1PSTEP" in line:
            break

        if line.startswith("-1"):
            # Clean off the leading structural marker (-1) so it doesn't get captured as a coordinate
            # We replace only the first occurrence of '-1'
            data_content = line.replace("-1", "", 1).strip()

            # Find all scientific numbers in the row
            floats_found = scientific_number_pattern.findall(data_content)

            # A valid line will yield exactly 3 floats: Ux, Uy, Uz
            if len(floats_found) == 3:
                try:
                    ux = float(floats_found[0])
                    uy = float(floats_found[1])
                    uz = float(floats_found[2])

                    # Extract the Node ID. It's whatever text is left at the front
                    # before the first scientific number starts.
                    node_id_part = data_content.split(floats_found[0])[0].strip()
                    node_id = int(node_id_part)

                    # Calculate total 3D resultant vector magnitude
                    magnitude = math.sqrt(ux ** 2 + uy ** 2 + uz ** 2)

                    if magnitude > max_magnitude:
                        max_magnitude = magnitude
                        max_node_id = node_id
                except (ValueError, IndexError):
                    continue  # Skip header labels or corrupted lines gracefully

    return max_node_id, max_magnitude

max_node, disp = extract_max_displacement(FRD)

print(max_node, disp)