import gmsh
import math


def run_crisp_pipeline(step_filepath, target_pt, output_inp_path, target_element_size=1.0):
    gmsh.initialize()
    gmsh.model.add("integrated_pipeline")

    # 1. Load the CAD file into OpenCASCADE
    gmsh.model.occ.importShapes(step_filepath)
    gmsh.model.occ.synchronize()

    # 2. Fix the touching bodies (PLC Intersect Error Fix)
    all_volumes = gmsh.model.getEntities(dim=3)
    gmsh.model.occ.fragment(all_volumes, [])
    gmsh.model.occ.synchronize()

    # 3. ANTI-BLOB ASSIGNMENT: Force Gmsh to register the 3D solid volumes
    # Without this, Gmsh only meshes the 2D surface skin and creates a blob.
    refreshed_volumes = gmsh.model.getEntities(dim=3)
    volume_tags = [tag for dim, tag in refreshed_volumes]
    # Creating a Physical Volume forces a full 3D tetrahedral fill
    gmsh.model.addPhysicalGroup(3, volume_tags, name="SOLID_BODIES")

    # 4. STRICT SIZING OVERRIDES: Disable automatic fallback bounds
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)  # Use our node sizes
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)  # Let it get smaller on curves
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)  # Force size inside the solid volume
    gmsh.option.setNumber("Mesh.MeshSizeMin", target_element_size * 0.1)  # Bound lower threshold
    gmsh.option.setNumber("Mesh.MeshSizeMax", target_element_size)  # Bound upper threshold
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 16)  # Smooth cylinder borders

    # Apply baseline sizing to all discrete structural points
    all_points = gmsh.model.getEntities(dim=0)
    gmsh.model.mesh.setSize(all_points, target_element_size)

    # 5. Locate the target cylinder face by its exact 3D center point
    surfaces = gmsh.model.getEntities(dim=2)
    matched_tag = None
    min_distance = float('inf')
    target_x, target_y, target_z = target_pt

    for dim, surface_tag in surfaces:
        if gmsh.model.getType(dim, surface_tag) == "Cylinder":
            x, y, z = gmsh.model.occ.getCenterOfMass(dim, surface_tag)
            distance = math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2 + (z - target_z) ** 2)

            if distance <= 0.5 and distance < min_distance:
                min_distance = distance
                matched_tag = surface_tag

    if matched_tag is None:
        gmsh.finalize()
        raise ValueError(f"No cylinder found at exact 3D point: {target_pt}")

    print(f"Verified Match! Surface ID {matched_tag} bound to location {target_pt}")

    # 6. Apply the Physical Group name for Abaqus constraints
    gmsh.model.addPhysicalGroup(2, [matched_tag], name="ROTATING_FACE_NODES")

    # 7. Generate 3D volume mesh and export
    try:
        gmsh.model.mesh.generate(3)
        gmsh.write(output_inp_path)
        print(f"Success! Crisp mesh saved to {output_inp_path}")
    except Exception as e:
        print(f"Meshing failed: {e}")
        gmsh.fltk.run()

    gmsh.finalize()


# --- Run the Complete System ---
my_target_coordinate = (3.0, 0.5, 23.7942)  # Your procedural target location
run_crisp_pipeline("../../MyFingerAutomationTestModel.step", my_target_coordinate, "simulation_ready.inp", target_element_size=1.5)

#(3.0, 0.5, 23.7942)
