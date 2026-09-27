import os
import shutil
from pathlib import Path
import zipfile

PROJECT_DIR = Path(__file__).resolve().parent.parent

def main():
    print("=" * 80)
    print("STEP 7: PACKAGING FINAL SUBMISSION (Vector Strike)")
    print("=" * 80)

    team_name = "Vector_Strike"
    zip_name = f"{team_name}_submission.zip"
    
    # Create temp directory for structure
    package_dir = PROJECT_DIR / f"{team_name}_submission"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()
    
    # 1. output/
    (package_dir / "output").mkdir()
    shutil.copy2(PROJECT_DIR / "output" / "matching_results.tsv", package_dir / "output" / "matching_results.tsv")
    shutil.copy2(PROJECT_DIR / "output" / "candidate_pairs.tsv", package_dir / "output" / "candidate_pairs.tsv")
    
    # 2. code/business_entity_resolution/src/
    code_dir = package_dir / "code" / "business_entity_resolution"
    src_dir = code_dir / "src"
    src_dir.mkdir(parents=True)
    
    for script in (PROJECT_DIR / "scripts").glob("*.py"):
        shutil.copy2(script, src_dir / script.name)
        
    # Write requirements.txt
    with open(code_dir / "requirements.txt", "w") as f:
        f.write("pandas==2.2.3\n")
        f.write("pyarrow==17.0.0\n")
        f.write("scikit-learn==1.5.2\n")
        f.write("lightgbm==4.5.0\n")
        f.write("tqdm\n")
        f.write("joblib\n")
        
    # Write README.md
    with open(code_dir / "README.md", "w") as f:
        f.write("# Vector Strike - Entity Resolution Pipeline\n\n")
        f.write("## Setup\n")
        f.write("pip install -r requirements.txt\n\n")
        f.write("## Running the Pipeline\n")
        f.write("Execute all scripts in numerical order:\n")
        f.write("python src/01_data_forensics.py\n")
        f.write("python src/02_preprocess.py\n")
        f.write("python src/03_blocking.py\n")
        f.write("python src/04_features.py\n")
        f.write("python src/05_train_model.py\n")
        f.write("python src/06_generate_submission.py\n\n")
        f.write("Or run the master script:\n")
        f.write("python src/run_pipeline.py\n")
        
    # 3. Documentation
    shutil.copy2(PROJECT_DIR / "Documentation_template.md", package_dir / "Documentation_template.md")
    
    # Zip it up
    print(f"Creating {zip_name}...")
    with zipfile.ZipFile(PROJECT_DIR / zip_name, 'w', zipfile.ZIP_BZIP2, compresslevel=9) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(package_dir)
                zipf.write(file_path, arcname)
                
    shutil.rmtree(package_dir)
    print(f"\nSUCCESS: Created {PROJECT_DIR / zip_name}")

if __name__ == "__main__":
    main()
