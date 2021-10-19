import os
import json
import argparse
import glob
from typing import Union
import sys
path_current_folder = os.path.dirname(os.path.abspath(__file__))
path_utils = os.path.abspath(os.path.join(path_current_folder, "../parser/utils"))
print(path_utils)
sys.path.append(path_utils)
from annotation_schema_utils import create_annotation_schema


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_folder",
    required=True,
    type=str,
    help="input folder containing the conlls",
)
parser.add_argument(
    "-o",
    "--output_path",
    type=str,
    help="directory to save the annotation schema. If not present, write in stdout",
)

args = parser.parse_args()
input_folder: str = args.input_folder
output_path: Union[str, None] = args.output_path

if os.path.isfile(input_folder):
    if input_folder.endswith(".conllu"):
        paths = [input_folder]
    else:
        raise BaseException("input file was not .conll neither a folder of conllu : ", input_folder)
else:
    paths = glob.glob(os.path.join(input_folder, "*.conllu"))
    if paths == []:
        raise BaseException("No conllu was found")

print("List of paths :", paths)
annotation_schema = create_annotation_schema(*paths)

if os.path.isdir(output_path):
    path_annotation_schema = os.path.join(output_path, "annotation_schema.json")
elif output_path.endswith(".json"):
        path_annotation_schema = output_path
else: 
    raise BaseException("output_path neither an existing folder or a json extension; output_path =", output_path)

if output_path: 
    # write annotaton schema
    with open(path_annotation_schema, "w") as output:
        json.dump(annotation_schema, output)
else: 
    # write in stdout
    print(annotation_schema)
