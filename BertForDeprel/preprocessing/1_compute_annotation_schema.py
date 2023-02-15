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
from annotation_schema_utils import get_annotation_schema_from_input_folder


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

annotation_schema = get_annotation_schema_from_input_folder(input_folder)

if output_path: 
    # write annotaton schema
    with open(output_path, "w") as output:
        json.dump(annotation_schema, output)
else: 
    # write in stdout
    print(annotation_schema)
