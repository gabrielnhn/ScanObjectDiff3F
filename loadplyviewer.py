import trimesh
from test_pc_scanobject import visualize_internal_labels
# trimesh.load("results_source.ply").show()
# trimesh.load("results_target_correspondence.ply").show()
# trimesh.load("final_ground_truth.ply").show()
# trimesh.load("./ground_truth_internal.ply").show()
# trimesh.load("./pointcloud1_with_features.ply").show()

# second_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/scene0271_00_00019.bin" 
# visualize_internal_labels(second_FILE)


trimesh.load("./final_source_gt.ply").show()
trimesh.load("./final_transfer_result.ply").show()

