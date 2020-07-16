import os
from subprocess import check_output

source = '/home/naivoder/hd/OID/Dataset/train'

for folder in os.listdir(source):
  target = f'{source}/{folder}'
  output = check_output(["python3", 'OIDv4_annotation_tool/OIDv4_to_VOC.py', "--sourcepath" , f"{source}/{folder}", "--dest_path", f"{target}"])
