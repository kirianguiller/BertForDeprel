import os
import json
import argparse
import glob
import conllu

# Return a dict containing "deprel", "upos" and "splitted_deprel"

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
def create_lemma_script_list(*paths):
    list_lemma = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as infile:
            result = conllu.parse(infile.read())

        for sequence in result:
            for token in sequence:
                list_lemma.append(token["lemma"])
    list_lemma.append("none")
    list_lemma = sorted(set(list_lemma))
    return list_lemma
                

def create_annotation_schema(*paths):  
    annotation_schema = {}  

    # Summary all deprels
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

    # Split deprels
    splitted_deprel = {}
    splitted_deprel["main"] = sorted(list(set(mains)))
    splitted_deprel["aux"] = sorted(list(set(auxs)))
    splitted_deprel["deep"] = sorted(list(set(deeps)))

    # Summary all pos
    upos = create_pos_list(*paths)

    # Summary all lemmas
    lemmas = create_lemma_script_list(*paths)

    annotation_schema["deprel"] = sorted(list(set(deprels)))
    annotation_schema["upos"] = sorted(upos)
    annotation_schema["splitted_deprel"] = splitted_deprel
    annotation_schema["lemmas"] = sorted(lemmas)
    print(annotation_schema)
    print(annotation_schema["lemmas"])
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
    required=True,
    type=str,
    help="directory to save the annotation schema",
)

args = parser.parse_args()
input_folder = args.input_folder
output_path = args.output_path

paths = glob.glob(os.path.join(input_folder, "*.conllu"))
if paths == []:
    raise BaseException("No conllu was found")

print("List of paths :", paths)
annotation_schema = create_annotation_schema(*paths)

with open(output_path, "w") as output:
    json.dump(annotation_schema, output)
