import torch
import cv2 as cv
import os
import numpy as np
l = os.listdir()

l = ["3.png", "4.png"]

device = torch.device("cuda")
dataset_path = "/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj"    
object = "horse.ply"
# object = "stanford-bunny.ply"

from pc_utils import load_ply_to_pytorch3d
partial_pcd = load_ply_to_pytorch3d(os.path.join(dataset_path, "indata", object))

# import torch
# import cv2 as cv
# import numpy as np
from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer, PerspectiveCameras, look_at_view_transform
from pytorch3d.ops import knn_points

def find_best_reference_pov_contour(pcd, device, pose_w=0.5, hole_w=10.0):
    points = pcd.points_padded()[0]
    total_points = points.shape[0]
    
    bbox = pcd.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bbox_center = (bbox_min + bbox_max) / 2.0
    distance = torch.sqrt(((bbox_max - bbox_min) ** 2).sum()) * 0.65

    # Raster settings
    image_size = 512
    raster_settings = PointsRasterizationSettings(
        image_size=image_size, 
        radius=0.02, 
        points_per_pixel=1
    )

    startv, endv = -80.0, 80.0
    starth, endh = -180.0, 180.0
    num = 40 
    batch_size = 16
    best_final_elev, best_final_azim = 0.0, 0.0

    for j in range(2):
        vers = torch.linspace(startv, endv, num, device=device)
        hors = torch.linspace(starth, endh, num, device=device)
        verss, horss = torch.meshgrid(vers, hors, indexing='ij')
        verss, horss = verss.flatten(), horss.flatten()
        
        best_loss = float('inf')
        best_elev, best_azim = 0.0, 0.0
        
        for i in range(0, len(verss), batch_size):
            chunk_elevs = verss[i:i+batch_size]
            chunk_azims = horss[i:i+batch_size]
            
            R, T = look_at_view_transform(dist=distance, elev=chunk_elevs, azim=chunk_azims, device=device, at=bbox_center.unsqueeze(0))
            cameras = PerspectiveCameras(device=device, R=R, T=T)
            rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
            
            pcd_batch = pcd.extend(len(chunk_elevs))
            fragments = rasterizer(pcd_batch)
            
            # --- Depth to OpenCV Integration ---
            # fragments.zbuf shape: (B, H, W, K)
            depth_maps = fragments.zbuf[..., 0] 
            idx_map = fragments.idx[..., 0] 
            cam_centers = cameras.get_camera_center() 
            
            for b in range(len(chunk_elevs)):
                # 1. Clean up Depth Map
                # Replace the background (-1) with a value further than the max depth
                d_map = depth_maps[b].clone()
                mask = (d_map == -1)
                max_d = d_map[~mask].max() if d_map[~mask].numel() > 0 else 1.0
                d_map[mask] = max_d * 1.2 # Push background to the rear
                
                # 2. Normalize to 8-bit for OpenCV
                d_min, d_max = d_map.min(), d_map.max()
                d_norm = (d_map - d_min) / (d_max - d_min + 1e-6)
                d_img = (d_norm.cpu().numpy() * 255).astype(np.uint8)
                
                # 3. Detect Depth Edges (Canny or Sobel)
                # This finds "holes" and sharp depth transitions
                edges = cv.Canny(d_img, 50, 150)
                
                # Use findContours on the edges to identify "closed" loops (holes)
                contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
                
                # Scoring: 
                # - More contours = more fragmentation/occlusion (Higher Loss)
                # - Total 'edge energy' = visual complexity
                edge_score = len(contours) 
                
                loss = edge_score
                
                if loss < best_loss:
                    best_loss = loss
                    best_elev = chunk_elevs[b].item()
                    best_azim = chunk_azims[b].item()
                    best_contours = contours
                    best_img = d_img
                    
                # if j == 1:
                    
            # Cleanup to prevent OOM
            del pcd_batch, fragments, cameras, rasterizer
            torch.cuda.empty_cache() 

        # Update search grid for next iteration
        interv = (endv - startv) / num
        interh = (endh - starth) / num
        startv, endv = best_elev - interv, best_elev + interv
        starth, endh = best_azim - interh, best_azim + interh
        best_final_elev, best_final_azim = best_elev, best_azim
        
        import random
        image = best_img
        for contour in contours:
            image = cv.drawContours(d_img, contour,-1, (random.randint(0,255),
                                        random.randint(0,255),
                                        random.randint(0,255)), 1)
        
        cv.imwrite(f"reference_test/contour-{j}-{chunk_elevs[b].item()}{chunk_azims[b].item()}.jpg",
                image)


    return best_final_elev, best_final_azim


a = find_best_reference_pov_contour(pcd=partial_pcd, device=device)


def find_best_reference_pov_compc(pcd, device, pose_w=0.5, hole_w=10.0):
    points = pcd.points_padded()[0]
    total_points = points.shape[0]
    
    bbox = pcd.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bbox_center = (bbox_min + bbox_max) / 2.0
    distance = torch.sqrt(((bbox_max - bbox_min) ** 2).sum()) * 0.65

    # Raster settings
    image_size = 512
    raster_settings = PointsRasterizationSettings(
        image_size=image_size, 
        radius=0.02, 
        points_per_pixel=1
    )

    startv, endv = -80.0, 80.0
    starth, endh = -180.0, 180.0
    num = 40 
    batch_size = 16
    best_final_elev, best_final_azim = 0.0, 0.0

    for j in range(2):
        vers = torch.linspace(startv, endv, num, device=device)
        hors = torch.linspace(starth, endh, num, device=device)
        verss, horss = torch.meshgrid(vers, hors, indexing='ij')
        verss, horss = verss.flatten(), horss.flatten()
        
        best_loss = float('inf')
        best_elev, best_azim = 0.0, 0.0
        
        for i in range(0, len(verss), batch_size):
            chunk_elevs = verss[i:i+batch_size]
            chunk_azims = horss[i:i+batch_size]
            
            R, T = look_at_view_transform(dist=distance, elev=chunk_elevs, azim=chunk_azims, device=device, at=bbox_center.unsqueeze(0))
            cameras = PerspectiveCameras(device=device, R=R, T=T)
            rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
            
            pcd_batch = pcd.extend(len(chunk_elevs))
            fragments = rasterizer(pcd_batch)
            
            # --- Depth to OpenCV Integration ---
            # fragments.zbuf shape: (B, H, W, K)
            depth_maps = fragments.zbuf[..., 0] 
            idx_map = fragments.idx[..., 0] 
            cam_centers = cameras.get_camera_center() 
            
            for b in range(len(chunk_elevs)):
                valid_mask = idx_map[b] != -1
                visible_indices = torch.unique(idx_map[b][valid_mask])
                
                if len(visible_indices) > 0:
                    # 1. Geometric Loss (Chamfer + Pose Distance)
                    visible_indices = visible_indices % total_points
                    visible_pts = points[visible_indices] 
                    dists, _, _ = knn_points(points.unsqueeze(0), visible_pts.unsqueeze(0), K=1)
                    fixed_cd = dists.squeeze().sqrt().mean() 
                    posedist = (cam_centers[b].unsqueeze(0) - points).square().sum(-1).sqrt().mean()
                    
                    # # 2. OpenCV Hole/Contour Detection
                    # # Create a binary mask from depth: 255 where points exist, 0 where background
                    depth_np = (valid_mask.cpu().numpy().astype(np.uint8)) * 255
                    
                    # # Find contours: RETR_EXTERNAL for silhouette, RETR_LIST for all holes
                    # contours, hierarchy = cv.findContours(depth_np, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
                    
                    # # Heuristic: We want a pose that sees the object clearly. 
                    # # If hole_count is high, it might indicate self-occlusion or gaps.
                    # # Or we penalize very fragmented views (too many separate contours).
                    # num_contours = len(contours)
                    # hole_penalty = max(0, num_contours - 1) # 1 contour is ideal (the object silhouette)

                    # Total Loss
                    loss = fixed_cd + (pose_w * posedist) 
                    # + (hole_w * hole_penalty)
                else:
                    loss = torch.tensor(float('inf'))
                
                if loss < best_loss:
                    best_loss = loss
                    best_elev = chunk_elevs[b].item()
                    best_azim = chunk_azims[b].item()
                    best_img = depth_np
            
            # Cleanup to prevent OOM
            del pcd_batch, fragments, cameras, rasterizer
            torch.cuda.empty_cache() 

        # Update search grid for next iteration
        interv = (endv - startv) / num
        interh = (endh - starth) / num
        startv, endv = best_elev - interv, best_elev + interv
        starth, endh = best_azim - interh, best_azim + interh
        best_final_elev, best_final_azim = best_elev, best_azim


        cv.imwrite(f"reference_test/compc-{j}-{chunk_elevs[b].item()}{chunk_azims[b].item()}.jpg",best_img)
    return best_final_elev, best_final_azim


find_best_reference_pov_compc(partial_pcd, device)

# for file in l:
#     if file.endswith(".py"):
#         continue

    
#     print(file)
#     image = cv.imread(file)
#     image = cv.resize(image, np.array(image.shape[:2][::-1])//2)
    
    
#     mask = cv.inRange(image, (0), (100))
#     # cv.imshow(file,mask)
#     contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
#     for contour in contours:
#         import random
#         image = cv.drawContours(image, contour,-1, (random.randint(0,255),
#                                                      random.randint(0,255),
#                                                      random.randint(0,255)), 1)
#     cv.imshow(file,image)
#     cv.waitKey(0)
        
    