import os
import glob
from typing import List, Set

from conllup.conllup import sentenceConllToJson, _featuresConllToJson, _featuresJsonToConll
from .lemma_script_utils import gen_lemma_script
from .types import AnnotationSchema_T


NONE_VOCAB = '_none' # default fallback


def compute_annotation_schema(*paths):
    all_sentences_json = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as infile:
            for sentence_conll in infile.read().split("\n\n"):
                if sentence_conll.strip():
                    all_sentences_json.append(sentenceConllToJson(sentence_conll))

    uposs: List[str] = []
    xposs: List[str] = []
    feats: List[str] = []
    deprels: List[str] = []
    lemma_scripts: List[str] = []
    for sentence_json in all_sentences_json:
        for token in sentence_json["treeJson"]["nodesJson"].values():
            deprels.append(token["DEPREL"])
            uposs.append(token["UPOS"])
            xposs.append(token["XPOS"])
            feats.append(_featuresJsonToConll(token["FEATS"]))

            lemma_script = gen_lemma_script(token["FORM"], token["LEMMA"])
            lemma_scripts.append(lemma_script)
    
    deprels.append(NONE_VOCAB)
    uposs.append(NONE_VOCAB)
    xposs.append(NONE_VOCAB)
    feats.append(NONE_VOCAB)
    lemma_scripts.append(NONE_VOCAB)

    deprels = sorted(set(deprels))
    uposs = sorted(set(uposs))
    xposs = sorted(set(xposs))
    feats = sorted(set(feats))
    lemma_scripts = sorted(set(lemma_scripts))

    annotation_schema: AnnotationSchema_T = {
        "deprels": deprels,
        "uposs": uposs,
        "xposs": xposs,
        "feats": feats,
        "lemma_scripts": lemma_scripts
    }
    return annotation_schema

def get_path_of_conllus_from_folder_path(path_folder: str):
    if os.path.isfile(path_folder):
        if path_folder.endswith(".conllu"):
            paths = [path_folder]
        else:
            raise BaseException("input file was not .conll neither a folder of conllu : ", path_folder)
    elif os.path.isdir(path_folder):
        paths = glob.glob(os.path.join(path_folder, "*.conllu"))
        if paths == []:
            raise BaseException("No conllu was found")
    else:
        raise Exception("No conllu was found (error 2)")
    return paths

def get_annotation_schema_from_input_folder(path_folder: str):
    path_conllus = get_path_of_conllus_from_folder_path(path_folder)
    annotation_schema = compute_annotation_schema(*path_conllus)
    return annotation_schema


def is_annotation_schema_empty(annotation_schema: AnnotationSchema_T):
    return (len(annotation_schema["uposs"]) == 0) or len(annotation_schema["deprels"]) == 0
