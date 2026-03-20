import os
import csv
from itertools import permutations
import torch
from label_transfer import run_transfer # Imports your modified function!

# --- CONFIG ---
RESULTS_DIR = "pc-feature-results-nodinoscore/"
OUTPUT_CSV = "transfer_results_nodinoscore.csv"

def get_base_names(class_dir):
    """Finds all unique object names by looking for .pt files"""
    bases = []
    for f in os.listdir(class_dir):
        if f.endswith(".pt"):
            # Strip the extension to get the base name
            bases.append(f.replace(".pt", ""))
    return bases

def main():
    print(f"Starting batch evaluation. Results will be saved to {OUTPUT_CSV}")
    
    # Open CSV in append mode so if it crashes, you don't lose data
    with open(OUTPUT_CSV, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write Header
        writer.writerow(["Class", "Source", "Target", "Exact_Accuracy", "Permutation_Accuracy"])
        
        classes = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
        
        for cls in classes:
            class_dir = os.path.join(RESULTS_DIR, cls)
            bases = get_base_names(class_dir)
            
            # Generate all pairs (A->B, B->A, etc.)
            pairs = list(permutations(bases, 2))
            print(f"\n--- Processing Class: {cls} ({len(bases)} objects, {len(pairs)} pairs) ---")
            
            for source_base, target_base in pairs:
                print(f"  Transfer: {source_base} -> {target_base}")
                
                # Construct paths
                s_feat = os.path.join(class_dir, source_base + ".pt")
                s_lbl  = os.path.join(class_dir, source_base + ".npy")
                t_feat = os.path.join(class_dir, target_base + ".pt")
                t_lbl  = os.path.join(class_dir, target_base + ".npy")
                
                with torch.no_grad():
                    exact, perm = run_transfer(s_feat, s_lbl, t_feat, t_lbl)
                
                if exact is not None:
                    # Save row to CSV immediately
                    writer.writerow([cls, source_base, target_base, f"{exact:.4f}", f"{perm:.4f}"])
                    # Force flush so data writes to disk instantly
                    csv_file.flush() 

if __name__ == "__main__":
    main()