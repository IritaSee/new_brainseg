import zipfile
import os

zip_filename = "DataBrainSeg.zip"

target_folder = "DataBrainSeg"

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    if os.path.isdir(target_folder):
        for root, dirs, files in os.walk(target_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start = target_folder)
                zipf.write(file_path, arcname)
        print(f"File {zip_filename} berhasil di buat")
    else:
        print(f"File {zip_filename} Tidak berhasil di buat")
        zipf.write(target_folder, os.path.basename(target_folder))
