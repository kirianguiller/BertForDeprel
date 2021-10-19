import os
import json
import argparse
import glob
from typing import Union
import conllu
import sys
path_current_folder = os.path.dirname(os.path.abspath(__file__))
path_utils = os.path.abspath(os.path.join(path_current_folder, "../parser/utils"))
print(path_utils)
sys.path.append(path_utils)
from lemma_script_utils import gen_lemma_rule


def create_lemma_script_list(*paths):
    lemma_scripts = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as infile:
            parsed_conllu = conllu.parse(infile.read())

        for sequence in parsed_conllu:
            for token in sequence:
                form = token["form"]
                lemma = token.get("lemma", "")
                lemma_script = "none"
                if lemma != "":
                    lemma_script = gen_lemma_rule(form, lemma)
                lemma_scripts.append(lemma_script)
    lemma_scripts.append("none")
    return sorted(list((set(lemma_scripts))))

def create_deprel_lists(*paths):
    deprels = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as infile:
            parsed_conllu = conllu.parse(infile.read())

        for sequence in parsed_conllu:
            for token in sequence:
                deprels.append(token["deprel"])
    return set(deprels)


def create_pos_list(*paths):
    list_pos = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as infile:
            result = conllu.parse(infile.read())

        for sequence in result:
            for token in sequence:
                list_pos.append(token["upostag"])
    list_pos.append("none")
    list_pos = sorted(set(list_pos))
    return list_pos


# TODO_LEMMA : add the create_lemma_script_list

def create_annotation_schema(*paths):
    annotation_schema = {}

    deprels = create_deprel_lists(*paths)

    mains, auxs, deeps = [], [], []

    for deprel in deprels:
        if deprel.count("@") == 1:
            deprel, deep = deprel.split("@")
            deeps.append(deep)
        if deprel.count(":") == 1:
            deprel, aux = deprel.split(":")
            auxs.append(aux)

        if (":" not in deprel) and ("@" not in deprel):
            mains.append(deprel)

    deprels = list(deprels)
    deprels.append("none")
    mains.append("none")
    auxs.append("none")
    deeps.append("none")

    splitted_deprel = {}
    splitted_deprel["main"] = sorted(list(set(mains)))
    splitted_deprel["aux"] = sorted(list(set(auxs)))
    splitted_deprel["deep"] = sorted(list(set(deeps)))

    upos = create_pos_list(*paths)
    lemma_scripts = create_lemma_script_list(*paths)
    annotation_schema["deprel"] = sorted(list(set(deprels)))
    annotation_schema["upos"] = sorted(upos)
    annotation_schema["splitted_deprel"] = splitted_deprel
    annotation_schema["lemma_script"] = lemma_scripts
    print(annotation_schema)
    return annotation_schema


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
