import numpy as np
import data_utils

def load_pc_file_with_colours(filename, suncg = False, with_bg = True):
    #load bin file
    pc=np.fromfile(filename, dtype=np.float32)
    # pc=np.fromfile(os.path.join(DATA_PATH, filename), dtype=np.float32)

    #first entry is the number of points
    #then x, y, z, nx, ny, nz, r, g, b, label, nyu_label
    if(suncg):
        pc = pc[1:].reshape((-1,3))
    else:
        pc = pc[1:].reshape((-1,11))

    positions = np.array(pc[:,0:3])
    colours = np.array(pc[:,6:9])
    return positions, colours

# SOURCE_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/014_00015.bin"
# TARGET_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/scene0271_00_00019.bin" 

TEST_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/014_00015.bin"
pc, colours = load_pc_file_with_colours(TEST_FILE)
data_utils.save_ply(pc, "./test.ply", colors=colours)

import trimesh
pc = trimesh.load("./test.ply")
pc.show()