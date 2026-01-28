import numpy as np

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

# import SurfaceAware3DFeaturesCode
# import scanobjectnn
# from scanobjectnn import data_utils
import data_utils



# TEST_FILE = "/home/gabrielnhn/diffscan/scanobjectnn/h5_files/main_split/test_objectdataset.h5"
# TEST_DATA, TEST_LABELS = data_utils.load_h5(TEST_FILE)
# pc = TEST_DATA[0]
# print(TEST_LABELS[0])


TEST_FILE = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/pillow/014_00015.bin"


pc, colours = load_pc_file_with_colours(TEST_FILE)

data_utils.save_ply(pc, "./test.ply", colors=colours)


import trimesh
pc = trimesh.load("./test.ply")
pc.show()