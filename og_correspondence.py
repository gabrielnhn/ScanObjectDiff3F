"""
Simple correspondence workflow from ipynb file
"""
import sys
sys.path.append("./Diff3F/")
# import packages


import torch
from diff3f_mesh import get_features_per_vertex
from utils import convert_mesh_container_to_torch_mesh, cosine_similarity, double_plot, get_colors, generate_colors
from time import time
from dataloaders.mesh_container import MeshContainer
from diffusion import init_pipe
from dino import init_dino
from functional_map import compute_surface_map

import meshplot as mp
mp.offline()


device = torch.device('cuda:0')
torch.cuda.set_device(device)
num_views = 2
H = 512
W = 512
num_images_per_prompt = 1
tolerance = 0.004
random_seed = 42
use_normal_map = True

def compute_features(device, pipe, dino_model, m, prompt):
    mesh = convert_mesh_container_to_torch_mesh(m, device=device, is_tosca=False)
    mesh_vertices = mesh.verts_list()[0]
    features = get_features_per_vertex(
        device=device,
        pipe=pipe, 
        dino_model=dino_model,
        mesh=mesh,
        prompt=prompt,
        mesh_vertices=mesh_vertices,
        num_views=num_views,
        H=H,
        W=W,
        tolerance=tolerance,
        num_images_per_prompt=num_images_per_prompt,
        use_normal_map=use_normal_map,
    )
    return features.cpu()

pipe = init_pipe(device)
dino_model = init_dino(device)

source_file_path = "Diff3F/meshes/cow.obj"
target_file_path = "Diff3F/meshes/camel.obj"
source_mesh = MeshContainer().load_from_file(source_file_path)
target_mesh = MeshContainer().load_from_file(target_file_path)

f_source = compute_features(device, pipe, dino_model, source_mesh, "cow")

f_target = compute_features(device, pipe, dino_model, target_mesh, "camel")

s = cosine_similarity(f_source.to(device),f_target.to(device))
s = torch.argmax(s, dim=0).cpu().numpy()
cmap_source = get_colors(source_mesh.vert); cmap_target = cmap_source[s]


double_plot(source_mesh,target_mesh,cmap_source,cmap_target)  