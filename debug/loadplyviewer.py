import trimesh
# import test_pc_scanobject

# trimesh.load("./final_source_gt.ply").show()
# trimesh.load("./final_transfer_result.ply").show()

# trimesh.load("./debug_heatmap_source.ply").show()
# trimesh.load("./debug_heatmap_target.ply").show()

# trimesh.load("debug_pca_visual1.ply").show()
# trimesh.load("debug_pca_visual2.ply").show()

# trimesh.load("debug_align_blue_original.ply").show()
# trimesh.load("debug_align_red_unprojected.ply").show()

# test_pc_scanobject.visualize_internal_labels(test_pc_scanobject.TEST_FILE1)

# trimesh.load("completed_shape_colored.ply").show()
# TEST_INDEX = 9
trimesh.load(f"GROUND_TRUTH_COMPLETE_SHAPE.ply").show()
trimesh.load(f"GROUND_TRUTH_PARTIAL_SHAPE.ply").show()

