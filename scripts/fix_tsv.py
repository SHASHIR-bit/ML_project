"""Fix TSV files: ensure LF line endings, UTF-8 no BOM, and repackage zip."""
import os, zipfile

OUTPUT_DIR = r"x:\ML_project\output"
PROJECT_DIR = r"x:\ML_project"

for fname in ["matching_results.tsv", "candidate_pairs.tsv"]:
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Rewrite with LF-only line endings
    with open(path, "wb") as f:
        for line in content.splitlines():
            f.write(line.encode("utf-8") + b"\n")
    
    size_mb = os.path.getsize(path) / 1e6
    print(f"Fixed {fname}: {size_mb:.2f} MB (LF endings, UTF-8)")

# Verify matching_results.tsv
with open(os.path.join(OUTPUT_DIR, "matching_results.tsv"), "rb") as f:
    chunk = f.read(200)
    has_crlf = b"\r\n" in chunk
    print(f"Verify CRLF gone: {not has_crlf}")
    print(f"First 100 bytes: {repr(chunk[:100])}")

# Repackage zip with standard DEFLATED compression (max level)
print("\nRepackaging zip with DEFLATED compression...")
zip_path = os.path.join(PROJECT_DIR, "Vector_Strike_submission.zip")

import shutil
from pathlib import Path

team_name = "Vector_Strike"
package_dir = Path(PROJECT_DIR) / f"{team_name}_submission"
if package_dir.exists():
    shutil.rmtree(package_dir)
package_dir.mkdir()

# 1. output/
(package_dir / "output").mkdir()
shutil.copy2(os.path.join(OUTPUT_DIR, "matching_results.tsv"), package_dir / "output" / "matching_results.tsv")
shutil.copy2(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), package_dir / "output" / "candidate_pairs.tsv")

# 2. code/business_entity_resolution/src/
code_dir = package_dir / "code" / "business_entity_resolution"
src_dir = code_dir / "src"
src_dir.mkdir(parents=True)

scripts_dir = Path(PROJECT_DIR) / "scripts"
for script in scripts_dir.glob("*.py"):
    shutil.copy2(script, src_dir / script.name)

with open(code_dir / "requirements.txt", "w") as f:
    f.write("pandas==2.2.3\npyarrow==17.0.0\nscikit-learn==1.5.2\nlightgbm==4.5.0\ntqdm\njoblib\n")

with open(code_dir / "README.md", "w") as f:
    f.write("# Vector Strike - Entity Resolution Pipeline\n\n")
    f.write("## Setup\npip install -r requirements.txt\n\n")
    f.write("## Running the Pipeline\nExecute all scripts in numerical order:\n")
    for i in range(1, 7):
        f.write(f"python src/{i:02d}_*.py\n")
    f.write("\nOr run: python src/run_pipeline.py\n")

# 3. Documentation
shutil.copy2(os.path.join(PROJECT_DIR, "Documentation_template.md"), package_dir / "Documentation_template.md")

# Zip with standard DEFLATED (compresslevel=9)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
    for root, dirs, files in os.walk(package_dir):
        for file in files:
            file_path = Path(root) / file
            arcname = file_path.relative_to(package_dir)
            zipf.write(file_path, arcname)

shutil.rmtree(package_dir)

size_mb = os.path.getsize(zip_path) / 1e6
print(f"Created {zip_path}: {size_mb:.2f} MB")
if size_mb > 512:
    print("WARNING: Still over 512 MB!")
else:
    print("OK: Under 512 MB limit")
