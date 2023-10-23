from pathlib import Path

from conllup.conllup import readConlluFile

from BertForDeprel.parser.utils.annotation_schema import (
    compute_annotation_schema,
    NONE_VOCAB,
    CONLLU_BLANK,
)

PATH_TEST_DATA_FOLDER = Path(__file__).parent / "data"
PATH_TEST_MODELS_FOLDER = Path(__file__).parent / "models"
PATH_TEST_CONLLU = PATH_TEST_DATA_FOLDER / "english.tiny.conllu"


def test_health():
    assert True


def test_compute_annotation_schema():
    sentences = readConlluFile(str(PATH_TEST_CONLLU))
    annotation_schema = compute_annotation_schema(
        sentences, relevant_miscs=["Subject"]
    )
    assert annotation_schema.deprels == [
        NONE_VOCAB,
        "comp:obj",
        "comp:pred",
        "compound",
        "det",
        "flat",
        "mod",
        "mod@relcl",
        "root",
        "subj",
        "udep",
    ]
    assert annotation_schema.feats == [
        "Degree=Sup",
        "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin",
        "NumType=Card",
        "Number=Sing",
        "PronType=Art",
        "PronType=Int,Rel",
        CONLLU_BLANK,
        NONE_VOCAB,
    ]

    assert annotation_schema.miscs == [
        "Subject=Dummy",
        CONLLU_BLANK,
        NONE_VOCAB,
    ]

    assert annotation_schema.uposs == [
        "ADJ",
        "ADP",
        "AUX",
        "DET",
        "NOUN",
        "NUM",
        "PRON",
        "PROPN",
        "VERB",
        NONE_VOCAB,
    ]
    assert annotation_schema.lemma_scripts == [
        NONE_VOCAB,
        "↑0;d¦",
        "↑0¦↓1;d¦",
        "↓0;abe",
        "↓0;d¦",
        "↓0;d¦-",
        "↓0;d¦----+y",
    ]
