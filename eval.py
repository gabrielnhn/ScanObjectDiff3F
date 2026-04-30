import torch
import numpy as np
import open3d as o3d
from pytorch3d.loss import chamfer_distance

class Metrics:
    def __init__(self, device='cuda'):
        self.device = device

    def calculate_metrics(self, p1, p2):
        """
        Calculates CD and EMD with the 10^2 scaling used in ComPC.pdf.
        """
        cd_l1_raw, _ = chamfer_distance(p1, p2, norm=1, point_reduction='mean')
        cd_l1 = cd_l1_raw * 100  # Scaling by 10^2

        return cd_l1.item()
        # , emd.item()

def resample_pcd(pcd_np, n_points=16384):
    """
    Standardizes point cloud resolution to 16,384 points[cite: 1].
    """
    if len(pcd_np) > n_points:
        idx = np.random.choice(len(pcd_np), n_points, replace=False)
        return pcd_np[idx]
    elif len(pcd_np) < n_points:
        idx = np.random.choice(len(pcd_np), n_points, replace=True)
        return pcd_np[idx]
    return pcd_np

def read_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    return np.array(pcd.points)

def evaluate_pair(res_path, gt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = Metrics(device=device)

    # Load and Resample to 16,384 points[cite: 1]
    out_np = resample_pcd(read_ply(res_path), n_points=16384)
    gt_np = resample_pcd(read_ply(gt_path), n_points=16384)

    # Convert to Tensors [B, N, 3]
    out_tensor = torch.tensor(out_np, dtype=torch.float32, device=device).unsqueeze(0)
    gt_tensor = torch.tensor(gt_np, dtype=torch.float32, device=device).unsqueeze(0)

    # cd, emd = metrics.calculate_metrics(out_tensor, gt_tensor)
    cd = metrics.calculate_metrics(out_tensor, gt_tensor)

    print(f"Metrics (Scaled by 10^2):")
    print(f"CD:  {cd:.4f}")
    # print(f"EMD: {emd:.4f}")

# Example Call:
# evaluate_pair("output_teapot.ply", "gt_teapot.ply")
if __name__ == '__main__':
    # Example usage for a single pair
    evaluate_pair(
        "polyscope-IMESH.ply",
        "/home/gabrielnhn/datasets/synthetic_redwood/upload/plyobj/gtdata/stanford-bunny.ply"
    )