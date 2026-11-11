import re
import math
import csv
import shutil
import subprocess
from pathlib import Path

CCX_EXE = r"C:\Users\DawnNightLight\Downloads\PrePoMax v2.5.0\Solver\ccx_dynamic.exe" # file to ccx solver

input_folder = Path("../../Misc/INP Files/FinrayLHSVariants")
output_file = Path("../../Misc/Run Results/FinrayLHSVariants.csv")
misc_output_folder = Path("../../Misc/Run Results/FinrayLHSVariants")

misc_output_folder.mkdir(parents=True, exist_ok=True)

input_list = [input_file for input_file in input_folder.iterdir()]

LOAD_NAME = "Concentrated_Force - 1"

LOADS = range(5, 55, 5)


pattern = re.compile(
    rf"({re.escape(LOAD_NAME)}\s*,\s*3\s*,\s*)"
    r"[-+]?\d*\.?\d+"
)

def extract_max_displacement(frd_file):
    with open(frd_file, 'r') as f:
        lines = f.readlines()
    pstep_indices = [idx for idx, line in enumerate(lines) if "1PSTEP" in line]

    if not pstep_indices:
        print("Error: No 1PSTEP markers found in the file. Check if simulation executed.")
        return None, None

    target_disp_idx = -1
    print(pstep_indices)
    for i in pstep_indices:
        if "DISP" in lines[i + 2]:
            print(i)
            disp_header_idx = i
            break
        else:
            return None, None

    max_magnitude = -1.0
    max_node_id = -1
    for idx in range(disp_header_idx + 1, len(lines)):
        line = lines[idx].strip()

        if line.startswith("-3") or "1PSTEP" in line:
            break

        if line.startswith("-1"):
            data_content = line.replace("-1", "", 1).strip()

            floats_found = scientific_number_pattern.findall(data_content)
            if len(floats_found) == 3:
                try:
                    ux = float(floats_found[0])
                    uy = float(floats_found[1])
                    uz = float(floats_found[2])
                    node_id_part = data_content.split(floats_found[0])[0].strip()
                    node_id = int(node_id_part)

                    magnitude = math.sqrt(ux ** 2 + uy ** 2 + uz ** 2)

                    if magnitude > max_magnitude:
                        max_magnitude = magnitude
                        max_node_id = node_id
                except (ValueError, IndexError):
                    continue  

    return max_node_id, max_magnitude

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["baseSize", "externalBeamWidth", "internalBeamSpacing", "internalBeamSize", "load", "max_Displacement"])

for variant in input_list:

    base_text = variant.read_text()
    variant_data = variant.stem.split("-")

    for load in LOADS:

        print(f"\n=== Running load {load} N ===")

        job_name = f"model_{load}"
        inp_file = Path(f"{job_name}.inp")

        new_text = pattern.sub(
            rf"\g<1>{-load}",
            base_text
        )

        inp_file.write_text(new_text)

        result = subprocess.run(
            [CCX_EXE, job_name],
            capture_output=True,
            text=True
        )

        print(result.stdout)
        print(result.stderr)

        result_folder = misc_output_folder / f"{variant.stem}-load-{load}"
        result_folder.mkdir(exist_ok=True)

        for file in Path("../../Code/Run").glob(f"{job_name}*"):
            shutil.move(str(file), result_folder / file.name)
            
        frd_file = result_folder / f"{job_name}.frd"

        max_disp = None

        if frd_file.exists():
            max_node, max_disp = extract_max_displacement(frd_file)
            print(f"Max displacement = {max_disp} at node {max_node}")
        else:
            print("No FRD file found")

        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(variant_data + [load, max_disp])

print("\nDONE — all loads completed.")
