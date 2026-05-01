import trimesh
import os
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
# trimesh.load(f"GROUND_TRUTH_COMPLETE_SHAPE.ply").show()
# trimesh.load(f"GROUND_TRUTH_PARTIAL_SHAPE.ply").show()

# trimesh.load(f"/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj/indata/horse.ply").show()
# trimesh.load(f"/home/gabrielnhn/LGM/workspace/gradio_output.ply").show()
# trimesh.load(f"final_instantmesh_shape.ply").show()
# trimesh.load(f"/home/gabrielnhn/LGM/workspace/white bunny/gradio_output.ply").show()


# trimesh.load(f"/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj/indata/horse.ply").show()

# trimesh.load(f"/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj/indata/stanford-bunny.ply").show()
# trimesh.load("IMESH.ply").show()
# trimesh.load("normalbunny-IMESH.ply").show()

# trimesh.load("debug_1_prediction_deterministic.ply").show()
# p1 = trimesh.load(f"final_instantmesh_shape.ply")


renders_dir = "./renders/horse/"
renders_dir = "./renders/cow/"
# renders_dir = "./renders/stanford-bunny/"
path1 = os.path.join(renders_dir, "debug_1_prediction_deterministic.ply")
path2 = os.path.join(renders_dir, "debug_2_ground_truth.ply")
path3 = os.path.join(renders_dir, "debug_3_partial_sensor.ply")

p1 = trimesh.load(path1)
p1.visual.vertex_colors = [255, 0, 0, 255] # Red

p2 = trimesh.load(path2)
p2.visual.vertex_colors = [0, 255, 0, 255] # Green

p3 = trimesh.load(path3)
p3.visual.vertex_colors = [0, 0, 255, 255] # Blue

# trimesh.Scene([p1, p2, p3]).show()
trimesh.Scene([p1, p3]).show()
# import os
# path = "/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj/indata/"
# # path = "/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj/gtdata/"
# l = os.listdir(path)
# for file in l:
#     filepath = os.path.join(path, file)
#     trimesh.load(filepath).show()
