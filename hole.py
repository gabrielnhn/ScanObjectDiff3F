import torch
import cv2 as cv
import os
import numpy as np
l = os.listdir()

l = ["3.png", "4.png"]

device = torch.device("cuda")
dataset_path = "/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj"    
# object = "horse.ply"
object = "stanford-bunny.ply"

partial_pcd = load_ply_to_pytorch3d(os.path.join(dataset_path, "indata", object))

def find_best_reference_pov(pcd, device, pose_w=0.5):
    """adapted from compc"""
    points = pcd.points_padded()[0]
    total_points = points.shape[0]
    
    bbox = pcd.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bbox_center = (bbox_min + bbox_max) / 2.0
    distance = torch.sqrt(((bbox_max - bbox_min) ** 2).sum()) * 0.65

    raster_settings = PointsRasterizationSettings(image_size=(128, 128), radius=0.03, points_per_pixel=1)

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
            idx_map = fragments.idx[..., 0] 
            cam_centers = cameras.get_camera_center() 
            
            for b in range(len(chunk_elevs)):
                valid_mask = idx_map[b] != -1
                visible_indices = torch.unique(idx_map[b][valid_mask])
                
                if len(visible_indices) > 0:
                    visible_indices = visible_indices % total_points
                    visible_points = points[visible_indices] 
                    dists, _, _ = knn_points(points.unsqueeze(0), visible_points.unsqueeze(0), K=1)
                    fixed_cd = dists.squeeze().sqrt().mean() 
                    cen = cam_centers[b]
                    posedist = (cen.unsqueeze(0) - points).square().sum(-1).sqrt().mean()
                    loss = fixed_cd + (pose_w * posedist)
                else:
                    loss = torch.tensor(float('inf'))
                
                if loss < best_loss:
                    best_loss = loss
                    best_elev = chunk_elevs[b].item()
                    best_azim = chunk_azims[b].item()
                    
            del pcd_batch, fragments, cameras, rasterizer
            torch.cuda.empty_cache() 

        interv = (endv - startv) / num
        interh = (endh - starth) / num
        startv, endv = best_elev - interv, best_elev + interv
        starth, endh = best_azim - interh, best_azim + interh
        best_final_elev, best_final_azim = best_elev, best_azim

    print(f"Optimal POV Found -> Azimuth: {best_final_azim:.1f}°, Elevation: {best_final_elev:.1f}°")
    return best_final_elev, best_final_azim


for file in l:
    if file.endswith(".py"):
        continue

    
    print(file)
    image = cv.imread(file)
    image = cv.resize(image, np.array(image.shape[:2][::-1])//2)
    
    
    mask = cv.inRange(image, (0), (100))
    # cv.imshow(file,mask)
    contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        import random
        image = cv.drawContours(image, contour,-1, (random.randint(0,255),
                                                     random.randint(0,255),
                                                     random.randint(0,255)), 1)
    cv.imshow(file,image)
    cv.waitKey(0)
        
    