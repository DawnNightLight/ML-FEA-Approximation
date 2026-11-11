from pathlib import Path
import numpy as np
import gmsh


def select_outermost_faces(surfaces, direction_vector, angle_tolerance_deg, geometric_tolerance, look_for_source=True,
                           coordinate_bounds=None):
    """
    Finds Gmsh surfaces facing a specific direction, applies an optional coordinate filter,
    isolates internal/hidden surfaces, and returns all structural node tags on the outer layer.

    Parameters:
    -----------
    surfaces : list
        List of (dim, tag) tuples from gmsh.model.getEntities(dim=2).
    direction_vector : list or np.array
        The 3D vector representing the target alignment direction.
    angle_tolerance_deg : float
        Angular variance tolerance in degrees.
    geometric_tolerance : float
        Numerical distance tolerance for filtering outermost flat planes.
    look_for_source : bool
        If True, targets the face closest to the force source (highest projection).
        If False, targets the opposite face (lowest projection).
    coordinate_bounds : tuple (int, list) or None
        Optional coordinate pre-filter. Example: (2, [10.0, 50.0]) means only consider
        surfaces whose center of mass Z-coordinate falls between 10.0 and 50.0.
        Set to None to disable.
    """
    target_vec = np.array(direction_vector) / np.linalg.norm(direction_vector)
    min_dot_product = np.cos(np.radians(angle_tolerance_deg))

    candidate_surfaces = []
    candidate_centroids = []
    matched_node_tags = set()

    # Looking for surfaces pointing in specific direction in specific region
    for dim, surface_tag in surfaces:
        x, y, z = gmsh.model.occ.getCenterOfMass(dim, surface_tag)
        centroid_coords = [x, y, z]
        try:
            if coordinate_bounds is not None:
                axis_index, val_range = coordinate_bounds
                current_coordinate = centroid_coords[axis_index]
                if not (val_range[0] <= current_coordinate <= val_range[1]):
                    continue

            normal = gmsh.model.getNormal(surface_tag, [x, y])
            normal_vec = np.array(normal)
            normal_vec = normal_vec / np.linalg.norm(normal_vec)

            alignment = np.dot(normal_vec, target_vec)

            if alignment >= min_dot_product:
                candidate_surfaces.append((dim, surface_tag))
                candidate_centroids.append(centroid_coords)
        except Exception:
            continue

    if not candidate_surfaces:
        return []

    candidate_centroids = np.array(candidate_centroids)

    projection_axis = -target_vec if look_for_source else target_vec
    projected_distances = np.dot(candidate_centroids, projection_axis)

    target_boundary_distance = np.max(projected_distances)

    for i, (dim, surface_tag) in enumerate(candidate_surfaces):
        if np.abs(projected_distances[i] - target_boundary_distance) < geometric_tolerance:
            node_tags, _, _ = gmsh.model.mesh.getNodes(dim, surface_tag, includeBoundary=True)
            matched_node_tags.update(node_tags)

    return sorted(list(matched_node_tags))


def run_automated_loading_pipeline(step_filepath, output_inp_path, force_direction=(1.0, 0.0, 0.0),
                                   angle_tolerance_deg=15.0, total_force_n=500.0, target_element_size=1.5):

    gmsh.initialize()
    gmsh.model.add("directional_loading_pipeline")

    gmsh.model.occ.importShapes(step_filepath)
    gmsh.model.occ.synchronize()

    all_volumes = gmsh.model.getEntities(dim=3)
    gmsh.model.occ.fragment(all_volumes, [])
    gmsh.model.occ.synchronize()

    refreshed_volumes = gmsh.model.getEntities(dim=3)
    volume_tags = [tag for dim, tag in refreshed_volumes]
    gmsh.model.addPhysicalGroup(3, volume_tags, name="SOLID_BODIES")

    all_points = gmsh.model.getEntities(dim=0)
    gmsh.model.mesh.setSize(all_points, target_element_size)

    gmsh.option.setNumber("Mesh.MeshSizeMin", target_element_size * 0.1)
    gmsh.option.setNumber("Mesh.MeshSizeMax", target_element_size)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 16)

    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 0)

    gmsh.model.mesh.generate(3)

    surfaces = gmsh.model.getEntities(dim=2)
    geometric_tolerance = target_element_size * 0.01

    z_search_window = (2, [300.0, 600.0])

    final_node_list = select_outermost_faces(
        surfaces=surfaces,
        direction_vector=force_direction,
        angle_tolerance_deg=angle_tolerance_deg,
        geometric_tolerance=geometric_tolerance,
        look_for_source=True,
        coordinate_bounds=z_search_window
    )

    if not final_node_list:
        gmsh.finalize()
        raise ValueError(f"Pipeline Error: No geometric nodes matched vector target orientation: {force_direction}")

    print(f"Extracted {len(final_node_list)} raw structural nodes matching direction criteria.")

    target_vec = np.array(force_direction)
    target_vec = target_vec / np.linalg.norm(target_vec)

    gmsh.write(output_inp_path)

    boundary_direction = [0.0, 0.0, -1.0]

    fixed_node_list = select_outermost_faces(
        surfaces=surfaces,
        direction_vector=boundary_direction,
        angle_tolerance_deg=angle_tolerance_deg,
        geometric_tolerance=geometric_tolerance,
        look_for_source=False
    )

    boundary_text_lines = []

    if fixed_node_list:
        boundary_text_lines.append("\n** -------------------------------------------------------------")
        boundary_text_lines.append("** BOUNDARY CONDITIONS DEFINED BY PIPELINE")
        boundary_text_lines.append("** -------------------------------------------------------------")

        boundary_text_lines.append("*Nset, nset=BoundarySet")

        for i in range(0, len(fixed_node_list), 16):
            chunk = fixed_node_list[i:i + 16]
            line_chunk = ", ".join(map(str, chunk))
            boundary_text_lines.append(line_chunk)

        boundary_text_lines.extend([
            "*Boundary, op=New",
            "** Name: Fixed-1",
            "*Boundary",
            "BoundarySet, 1, 6, 0."
        ])

        boundary_deck_output = "\n".join(boundary_text_lines)

        with open(output_inp_path, "a") as file_append:
            file_append.write(boundary_deck_output + "\n")

        print(
            f"Successfully injected node set definition & constraints for {len(fixed_node_list)} boundary nodes into {output_inp_path}")
    else:
        print("Warning: No boundary nodes were matched for fixity. INP file left unmodified.")

    if fixed_node_list:
        boundary_text_lines.append("*Nset, nset=BoundarySet")

        for i in range(0, len(fixed_node_list), 16):
            chunk = fixed_node_list[i:i + 16]
            line_chunk = ", ".join(map(str, chunk))
            boundary_text_lines.append(line_chunk)

        boundary_text_lines.extend([
            "*Boundary, op=New",
            "** Name: Fixed-1",
            "*Boundary",
            "BoundarySet, 1, 6, 0."
        ])

        boundary_deck_output = "\n".join(boundary_text_lines)
        print(f"Grouped {len(fixed_node_list)} boundary nodes into set 'BoundarySet'")
    else:
        print("Warning: No boundary nodes were matched for fixity.")

    gmsh.finalize()

    with open(output_inp_path, "a") as inp:
        inp.write("\n** " + "=" * 40 + "\n")
        inp.write("** AUTOMATED DIRECTIONAL NODE SET (NO EXTRA SHELL ELEMENTS)\n")
        inp.write("** " + "=" * 40 + "\n")
        inp.write("*NSET, NSET=LOAD_SURFACE_NODES\n")

        for i in range(0, len(final_node_list), 10):
            chunk = final_node_list[i:i + 10]
            line = ", ".join(map(str, chunk))
            inp.write(line + "\n")

        inp.write("** Materials + ++++++++++++++++++++++++++++++++++++++++++++++\n")
        inp.write("**\n")
        inp.write("*Material, Name = main\n")
        inp.write("*Elastic\n")
        inp.write("10, 0.45\n")
        inp.write("**\n")
        inp.write("** Sections + +++++++++++++++++++++++++++++++++++++++++++++++\n")
        inp.write("**\n")
        inp.write("*Solid section, Elset = Volume1, Material = main\n")

        inp.write("\n** " + "=" * 40 + "\n")
        inp.write("** LOADING STEP DECLARATION\n")
        inp.write("** " + "=" * 40 + "\n")
        inp.write("*Step, Nlgeom, Inc=10\n")
        inp.write("*Static, Solver=Pardiso\n")
        inp.write("0.001, 1, 1E-05, 1E+30\n")

        inp.write("*Boundary, op=new")
        inp.write("** Name: Fixed-1\n")
        inp.write("*Boundary\n")
        inp.write("BoundarySet, 1, 6, 0.\n")

        #print(f"Extracted {len(fixed_node_tags)} boundary nodes at Z={lowest_z:.4f}")

        inp.write("** Exert vector forces smoothly mapped to target degrees of freedom\n")
        inp.write("*Cload, op=New\n")
        inp.write("*Dload, op=New\n")
        inp.write("** Name: Concentrated_Force - 1\n")
        inp.write("*Cload\n")

        if abs(target_vec[0]) > 0.001:
            inp.write(f"LOAD_SURFACE_NODES, 1, {total_force_n * target_vec[0]:.2f}\n")
        if abs(target_vec[1]) > 0.001:
            inp.write(f"LOAD_SURFACE_NODES, 2, {total_force_n * target_vec[1]:.2f}\n")
        if abs(target_vec[2]) > 0.001:
            inp.write(f"LOAD_SURFACE_NODES, 3, {total_force_n * target_vec[2]:.2f}\n")

        inp.write("*Node file\n")
        inp.write("RF, U\n")
        inp.write("*El file\n")
        inp.write("S, E, NOE\n")
        inp.write("**\n")
        inp.write("** End step ++++++++++++++++++++++++++++++++++++++++++++++++\n")
        inp.write("**\n")
        inp.write("*End step\n")

    print(f"Success! Finished writing assembly simulation file layout to: {output_inp_path}")


#EXECUTION LOOP 
input_folder = Path("../../Misc/STEP Files/FinrayLHSVariants")
output_folder = Path("../../Misc/INP Files/FinrayLHSVariants")


output_folder.mkdir(parents=True, exist_ok=True)

failed = []
for input_file in input_folder.iterdir():
    if not input_file.is_file():
        continue

    output_file = output_folder / f"{input_file.stem}.inp"

    print("Input:", input_file)
    print("Output:", output_file)
    try:
        if not output_file in output_folder.iterdir():
            run_automated_loading_pipeline(
                step_filepath=str(input_file),
                output_inp_path=str(output_file),
                force_direction=(-1.0, 0.0, 0.0),  # Target loading orientation vector
                angle_tolerance_deg=20.0,  # Acceptable angle deviation
                total_force_n=50.0,  # Force of load
                target_element_size=50  # Element size in mm
            )
    except:
        print(f"{input_file.stem} failed.")
        failed.append(input_file.stem)

    print(f"Processed: {input_file.name} -> {output_file.name}")

print(failed)

# ['265-79-28-32', '266-68-23-31', '287-86-23-31', '287-99-24-31', '297-88-31-22', '298-95-30-22', '321-96-29-35', '322-73-30-27', '332-98-34-31', '333-41-27-35']
# 325-94-23-32 meshed with non-positive Jacobian determinants
# 251-75-23-28 and 258-77-24-33 both failed and were moved into the LHSFailed folder