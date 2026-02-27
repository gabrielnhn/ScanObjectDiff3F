import numpy as np
import trimesh

TEST_FILE1 = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/sofa/294_00002.bin"
TEST_FILE2 = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/sofa/080_00003.bin"
TEST_FILE3 = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/toilet/scene0447_00_00006.bin"
TEST_FILE4 = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/sofa/scene0162_00_00003.bin"
TEST_FILE5 = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/014_00015.bin"
TEST_FILE6 = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/scene0219_00_00003.bin"
TEST_FILE7 = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/scene0362_00_00010.bin"
TEST_FILE8 = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/toilet/scene0153_00_00006.bin"
TEST_FILE9 = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/toilet/scene0447_00_00006.bin"

TEST_FILE10 = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/sink/scene0265_00_00011.bin"
TEST_FILE11 = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/sink/scene0399_00_00002.bin"
TEST_FILE12 = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/sink/scene0434_00_00010.bin"
TEST_FILE13 = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/sofa/scene0001_00_00003.bin"

TEST_FILE14 = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/sofa/scene0033_00_00011.bin"

TEST_FILE15 = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/sofa/scene0392_00_00008.bin"

def visualize_internal_labels(bin_path):
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

    labels = col_10.astype(int)

    colors = np.zeros((n_points, 3), dtype=np.uint8) + 128 # Grey Base
    
    unique_lbls = np.unique(labels)
    palette = np.array([
        [255, 0, 0],  
        [0, 255, 0],  
        [0, 0, 255],  
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
    ])
    
    for i, lbl in enumerate(unique_lbls):
        safe_idx = lbl if lbl >= 0 else len(palette) - 1 
        color = palette[safe_idx % len(palette)] 
        
        colors[labels == lbl] = color
        print(f"  Class {lbl} -> {color}")

    pcd = trimesh.PointCloud(positions, colors=rgb_colors/255)
    pcd.show()
    pcd = trimesh.PointCloud(positions, colors=colors)
    pcd.show()


if __name__ == "__main__":
    visualize_internal_labels(TEST_FILE1)
    visualize_internal_labels(TEST_FILE2)
    # visualize_internal_labels(TEST_FILE3)
    # visualize_internal_labels(TEST_FILE4)
    # visualize_internal_labels(TEST_FILE5)
    # visualize_internal_labels(TEST_FILE6)
    # visualize_internal_labels(TEST_FILE7)
    # visualize_internal_labels(TEST_FILE8)
    # visualize_internal_labels(TEST_FILE9)
    # visualize_internal_labels(TEST_FILE10)
    # visualize_internal_labels(TEST_FILE11)
    # visualize_internal_labels(TEST_FILE12)
    # visualize_internal_labels(TEST_FILE13)
    # visualize_internal_labels(TEST_FILE14)
    # visualize_internal_labels(TEST_FILE15)