import sys
sys.path.append("./Diff3F/")

import torch
import numpy as np
import trimesh

# from diff3f_pc import get_features_per_point
# from diff3f_depthonly import get_features_per_point
from diff3f_dinoonly import get_features_per_point
from pc_utils import load_scanobjectnn_to_pytorch3d, save_pointcloud_with_features
from utils import cosine_similarity, get_colors
from diffusion import init_pipe
from dino2 import init_dino


device = torch.device('cuda:0')
torch.cuda.set_device(device)

# num_views = 20
num_views = 50
# num_views = 1
H = 512
W = 512
num_images_per_prompt = 1
tolerance = 0.1
use_normal_map = True

def compute_pc_features_dinoonly(device,
                        # pipe,
                        dino_model, pcd,
                        # prompt
                        ):
    features = get_features_per_point(
        device=device,
        # pipe=pipe, 
        dino_model=dino_model,
        pcd=pcd, 
        # prompt=prompt,
        num_views=num_views,
        H=H,
        W=W,
        tolerance=tolerance,
        # num_images_per_prompt=num_images_per_prompt,
        # use_normal_map=use_normal_map,
    )# .cpu()
    return features

def compute_pc_features_diff(device,
                        pipe,
                        dino_model, pcd,
                        prompt
                        ):
    features = get_features_per_point(
        device=device,
        pipe=pipe, 
        dino_model=dino_model,
        pcd=pcd, 
        prompt=prompt,
        num_views=num_views,
        H=H,
        W=W,
        tolerance=tolerance,
        num_images_per_prompt=num_images_per_prompt,
        use_normal_map=use_normal_map,
    )# .cpu()
    return features


# pipe = init_pipe(device)
dino_model = init_dino(device)

# first_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/014_00015.bin"
# second_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/scene0271_00_00019.bin" 
first_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/sofa/080_00003.bin"

# first_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/sofa/294_00002.bin"
# f_first = compute_pc_features_diff(device, pipe, dino_model, first_pcd, "a pillow")



first_pcd, first_labels = load_scanobjectnn_to_pytorch3d(first_FILE, device)
print("computing features for first (pillow a)...")
f_first = compute_pc_features_dinoonly(device, dino_model, first_pcd)
# save_pointcloud_with_features(first_pcd, f_first, "pointcloud1_with_features", first_labels)
save_pointcloud_with_features(first_pcd, f_first, "pointcloud2_with_features", first_labels)


del f_first, first_pcd, first_labels
torch.cuda.empty_cache()
import gc
gc.collect()

second_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/sofa/294_00002.bin"
second_pcd, second_labels = load_scanobjectnn_to_pytorch3d(second_FILE, device)




print("computing features for second (pillow b)...")
f_second = compute_pc_features_dinoonly(device, dino_model, second_pcd)
save_pointcloud_with_features(second_pcd, f_second, "pointcloud2_with_features", second_labels)
