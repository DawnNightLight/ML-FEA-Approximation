import re
import math
import csv
import shutil
import subprocess
from pathlib import Path

# ==========================================================
# USER SETTINGS
# ==========================================================

CCX_EXE = r"C:\Users\murra\Downloads\PrePoMax v2.5.0\Solver\ccx_dynamic.exe"

inp_list = [Path("SlantedTallOutwardsUltraSoft.inp"), Path("SlantedTallOutwardsUltraHard.inp"),
            Path("SlantedTallInwardsUltraSoft.inp"), Path("SlantedTallInwardsUltraHard.inp"),
Path("SlantedTallOutwardsSoftSurface.inp"), Path("SlantedTallOutwardsHardSurface.inp"),
            Path("SlantedTallInwardsSoftSurface.inp"), Path("SlantedTallInwardsHardSurface.inp"),
            Path("SlantedTallOutwardsNylon.inp"), Path("SlantedTallInwardsNylon.inp")]
inp_list = [Path("SlantedTallOutwardsNylon.inp"), Path("SlantedTallInwardsNylon.inp")]

LOAD_NAME = "Internal_Selection-1_Concentrated_Force-1"

LOADS = range(5, 55, 5)
LOADS = range(3, 14, 1)

result_dir_list = [Path("MaterialChange\TrapezioidSoft"), Path("MaterialChange\TrapezioidHard"),
                   Path("MaterialChange\RhomboidSoft"), Path("MaterialChange\RhomboidHard"),
                    Path("MaterialChange\TrapezioidSoftSurface"), Path("MaterialChange\TrapezioidHardSurface"),
                   Path("MaterialChange\RhomboidSoftSurface"), Path("MaterialChange\RhomboidHardSurface"),
                   Path("MaterialChange\RhomboidNylon"), Path("MaterialChange\TrapezioidNylon")]
result_dir_list = [Path("MaterialChange\RhomboidNylonRegion"), Path("MaterialChange\TrapezioidNylonRegion")]

csv_file = ["results_trapezioid_soft.csv", "results_trapezioid_hard.csv",
            "results_rhomboid_soft.csv", "results_rhomboid_hard.csv",
            "results_trapezioid_soft_surface.csv", "results_trapezioid_hard_surface.csv",
            "results_rhomboid_soft_surface.csv", "results_rhomboid_hard_surface.csv",
            "results_trapezioid_nylon.csv", "results_rhomboid_nylon.csv"]
csv_file = ["results_trapezioid_nylon_region.csv", "results_rhomboid_nylon_region.csv"]

# ==========================================================
# LOAD REPLACEMENT PATTERN
# ==========================================================

pattern = re.compile(
    rf"({re.escape(LOAD_NAME)}\s*,\s*3\s*,\s*)"
    r"[-+]?\d*\.?\d+"
)


# ==========================================================
# FRD PARSER (MAX DISPLACEMENT)
# ==========================================================

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
        if "DISP" in lines[i + 2]:
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


for SomeInput in range(len(inp_list)):
    BASE_INP = inp_list[SomeInput]
    RESULTS_DIR = result_dir_list[SomeInput]
    RESULTS_DIR.mkdir(exist_ok=True)

    CSV_FILE = RESULTS_DIR / csv_file[SomeInput]
    # ==========================================================
    # READ BASE INPUT
    # ==========================================================

    base_text = BASE_INP.read_text()

    # ==========================================================
    # CSV INIT
    # ==========================================================

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Load", "Max_Displacement"])

    # ==========================================================
    # MAIN LOOP
    # ==========================================================

    for load in LOADS:

        print(f"\n=== Running load {load} N ===")

        job_name = f"model_{load}"
        inp_file = Path(f"{job_name}.inp")

        # ------------------------------------------------------
        # Modify load
        # ------------------------------------------------------

        new_text = pattern.sub(
            rf"\g<1>{-load}",
            base_text
        )

        inp_file.write_text(new_text)

        # ------------------------------------------------------
        # Run CalculiX
        # ------------------------------------------------------

        result = subprocess.run(
            [CCX_EXE, job_name],
            capture_output=True,
            text=True
        )

        print(result.stdout)
        print(result.stderr)

        # ------------------------------------------------------
        # Create result folder
        # ------------------------------------------------------

        result_folder = RESULTS_DIR / f"Load_{load}"
        result_folder.mkdir(exist_ok=True)

        # ------------------------------------------------------
        # Move ALL related files
        # ------------------------------------------------------

        for file in Path("../..").glob(f"{job_name}*"):
            shutil.move(str(file), result_folder / file.name)

        # ------------------------------------------------------
        # Extract max displacement
        # ------------------------------------------------------

        frd_file = result_folder / f"{job_name}.frd"

        max_disp = None

        if frd_file.exists():
            max_node, max_disp = extract_max_displacement(frd_file)
            print(f"Max displacement = {max_disp} at node {max_node}")
        else:
            print("No FRD file found")

        # ------------------------------------------------------
        # Save to CSV
        # ------------------------------------------------------

        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([load, max_disp])

print("\nDONE — all loads completed.")
