import numpy as np
import trimesh

# TEST_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/014_00015.bin"
# TEST_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/scene0219_00_00003.bin"
# TEST_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/scene0362_00_00010.bin"
# TEST_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/toilet/scene0153_00_00006.bin"
TEST_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/toilet/scene0447_00_00006.bin"


def visualize_internal_labels(bin_path):
    # 1. Load Data
    raw_geom = np.fromfile(bin_path, dtype=np.float32)
    n_points = int(raw_geom[0])
    
    # Reshape (N, 11) -> Skip header
    data = raw_geom[1:].reshape((n_points, 11))
    
    positions = data[:, 0:3]
    rgb_colors = data[:, 6:9]
    # Check the last two columns
    col_9 = data[:, 9]  # Index 9 (10th column)
    col_10 = data[:, 10] # Index 10 (11th column)
    
    print(f"Checking internal columns for {n_points} points...")
    print(f"Col 9 Unique: {len(np.unique(col_9))} values")
    print(f"Col 10 Unique: {len(np.unique(col_10))} values -> {np.unique(col_10)}")

    # 2. Select the likely label column
    # We want small integers (0, 1, 2)
    if len(np.unique(col_10)) < 20:
        labels = col_10.astype(int)
        print(">>> Using Column 10 as Labels")
    elif len(np.unique(col_9)) < 20:
        labels = col_9.astype(int)
        print(">>> Using Column 9 as Labels")
    else:
        print("No obvious label column found inside .bin file.")
        return

    # 3. Visualize
    colors = np.zeros((n_points, 3), dtype=np.uint8) + 128 # Grey Base
    
    unique_lbls = np.unique(labels)
    # Simple Palette: Red, Green, Blue, Yellow
    palette = [
        [255, 0, 0], 
        [0, 255, 0], 
        [0, 0, 255], 
        [255, 255, 0]
        [0, 255, 255]
        [255, 0, 255]
    ]
    
    for i, lbl in enumerate(unique_lbls):
        # Skip background if it's 0 or -1? 
        # Usually 0 is a class in ScanObjectNN, but let's see.
        # Just map everything to a color for now.
        color = palette[i % len(palette)]
        colors[labels == lbl] = color
        print(f"  Class {lbl} -> {color}")

    pcd = trimesh.PointCloud(positions, colors=colors)
    # pcd.export("ground_truth_internal.ply")
    # print("Saved 'ground_truth_internal.ply'")
    pcd.show()
    pcd = trimesh.PointCloud(positions, colors=rgb_colors/255)
    pcd.show()

visualize_internal_labels(TEST_FILE)