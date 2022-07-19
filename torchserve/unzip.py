import zipfile

with zipfile.ZipFile('DCNv2_latest.zip', 'r') as zip_ref:
    zip_ref.extractall('DCNv2_latest')