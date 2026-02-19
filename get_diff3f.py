import sys
sys.path.append("./Diff3F/")

import torch
import numpy as np
import trimesh

# from diff3f_pc import get_features_per_point
from diff3f_depthonly import get_features_per_point
from pc_utils import load_scanobjectnn_to_pytorch3d, save_pointcloud_with_features
from utils import cosine_similarity, get_colors
from diffusion import init_pipe
from dino import init_dino




device = torch.device('cuda:0')
torch.cuda.set_device(device)

# num_views = 20
num_views = 50
# num_views = 1
H = 512
W = 512
num_images_per_prompt = 1
tolerance = 0.004
use_normal_map = True

def compute_pc_features(device, pipe, dino_model, pcd, prompt):
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

def save_ply_with_colors(points, colors, filename):
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    
    pcd = trimesh.PointCloud(vertices=points, colors=colors)
    pcd.export(filename)
    print(f"Saved {filename}")

pipe = init_pipe(device)
dino_model = init_dino(device)


import os
base_dir = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/"
classes = []
for file in os.listdir(base_dir):
    if os.path.isdir(os.path.join(base_dir, file)):
        classes.append(file)
    # else:
    #     print(file)


all_objects = {obj: [] for obj in classes}

for obj in classes:
    # print(obj)
    class_path = os.path.join(base_dir, obj)
    # print(class_path)
    for root, dirs, files in os.walk(class_path):
        if not dirs:
            dirs = [""]
        for dir in dirs:
            # print(dir)
            for file in files:
                filename = os.path.join(root, dir, file)
                
                if not filename.endswith(".bin"):
                    continue
                if "indices" in filename:
                    continue
                
                if "_part" in file:
                    continue
                
                # print("file", filename)
                all_objects[obj].append(filename)

# print("NUMBER OF POINT CLOUDS", sum([len(all_objects[cl]) for cl in all_objects]) )

for c in all_objects:
    c_path = os.path.join("results/", c)
    if not os.path.exists(c_path):
        os.mkdir(c_path)
        
    for filename in all_objects[c]:
        basename = os.path.basename(filename)
        
        already_computed = False
        for others in os.listdir(c_path):
            if basename in others:
                already_computed = True

        if already_computed:
            continue        
        
        destination_filename = os.path.join(c_path, basename)
        print(destination_filename)
        
        pcd, labels = load_scanobjectnn_to_pytorch3d(filename, device)
        f_first = compute_pc_features(device, pipe, dino_model, pcd, c)
        save_pointcloud_with_features(pcd, f_first, 
                                      destination_filename,
                                      labels)
        del pcd
        del labels
        del f_first

