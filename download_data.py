import os
import subprocess

def download_data():
    BUCKET = "gs://waymo-motion-pipeline-ashonfire/processed_enriched"
    LOCAL_DIR = "./data/processed_enriched"
    
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    print("Downloading 55 Batches (Train + Eval) from GCS...")
    
    # Use gsutil -m for parallel download
    # We download batch_00 to batch_99 (Full Dataset)
    for i in range(100):
        batch = f"batch_{i:02d}"
        gcs_path = f"{BUCKET}/{batch}"
        local_path = f"{LOCAL_DIR}/{batch}"
        
        if os.path.exists(local_path):
            print(f"Skipping {batch} (Already exists)")
            continue
            
        print(f"Downloading {batch}...")
        try:
            # Recursive copy of the batch directory
            subprocess.check_call(["gsutil", "-m", "cp", "-r", gcs_path, LOCAL_DIR])
        except Exception as e:
            print(f"Failed to download {batch}: {e}")

if __name__ == "__main__":
    download_data()
