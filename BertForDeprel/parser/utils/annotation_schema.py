from dataclasses import dataclass, field
from typing import Any, List, Mapping

from conllup.conllup import _featuresJsonToConll, readConlluFile

from .lemma_script_utils import gen_lemma_script

NONE_VOCAB = "_none"  # default fallback


@dataclass
class AnnotationSchema_T:
    deprels: List[str] = field(default_factory=list)
    uposs: List[str] = field(default_factory=list)
    xposs: List[str] = field(default_factory=list)
    feats: List[str] = field(default_factory=list)
    lemma_scripts: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(schema_dict: Mapping[str, Any]):
        annotation_schema = AnnotationSchema_T()
        # TODO: Check the validity of this first; at least a version number
        annotation_schema.__dict__.update(schema_dict)
        return annotation_schema

    def is_empty(self):
        return not (self.uposs or self.deprels)


def compute_annotation_schema(*paths):
    all_sentences_json = []
    for path in paths:
        all_sentences_json.extend(readConlluFile(path))

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

    annotation_schema = AnnotationSchema_T(
        deprels=deprels,
        uposs=uposs,
        xposs=xposs,
        feats=feats,
        lemma_scripts=lemma_scripts,
    )
    return annotation_schema
