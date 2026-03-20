from compute_features_pair import *




import os
base_dir = "/home/gabrielnhn/datasets/object_dataset_complete_with_parts/"
classes = []
# for file in os.listdir(base_dir):
#     if os.path.isdir(os.path.join(base_dir, file)):
#         classes.append(file)
    # else:
    #     print(file)

classes = ["sofa"]


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

print("NUMBER OF POINT CLOUDS", sum([len(all_objects[cl]) for cl in all_objects]) )

base_results_dir = "pc-feature-results-yesdinoscore/"
if not os.path.exists(base_results_dir):
    os.mkdir(base_results_dir)


for c in all_objects:
    c_path = os.path.join(base_results_dir, c)
    if not os.path.exists(c_path):
        os.mkdir(c_path)
        
    for filename in all_objects[c]:
        basename = os.path.basename(filename)
        
        # 1. Strip the .bin extension
        name_without_ext = basename.replace(".bin", "")
        
        # 2. Check exactly if the .pt file already exists
        expected_pt_file = os.path.join(c_path, name_without_ext + ".pt")
        
        if os.path.exists(expected_pt_file):
            print(f"Skipping (already computed): {name_without_ext}")
            continue        
        
        if "scene0315_00_00003" in basename:
            continue
        
        destination_filename = os.path.join(c_path, basename)
        print(f"Computing: {destination_filename}")
        
        pcd, labels = load_scanobjectnn_to_pytorch3d(filename, device)
        f_first = compute_pc_features_dinoonly(device, dino_model, pcd, True)
        save_pointcloud_with_features(pcd, f_first, 
                                      destination_filename,
                                      labels)
        del pcd
        del labels
        del f_first
        torch.cuda.empty_cache()
        import gc
        gc.collect()
