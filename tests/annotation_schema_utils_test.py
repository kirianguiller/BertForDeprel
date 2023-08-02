from pathlib import Path

from BertForDeprel.parser.utils.annotation_schema_utils import compute_annotation_schema

PATH_TEST_DATA_FOLDER = Path(__file__).parent / "data"
PATH_TEST_MODELS_FOLDER = Path(__file__).parent / "models"
PATH_TEST_CONLLU = str(PATH_TEST_DATA_FOLDER / "english.conllu")


def test_health():
    assert True


def test_compute_annotation_schema():
    annotation_schema = compute_annotation_schema(PATH_TEST_CONLLU)
    assert annotation_schema.deprels == [
        "_none",
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
        "_",
        "_none",
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
        "_none",
    ]
    assert annotation_schema.lemma_scripts == [
        "_none",
        "↑0;d¦",
        "↑0¦↓1;d¦",
        "↓0;abe",
        "↓0;d¦",
        "↓0;d¦-",
        "↓0;d¦----+y",
    ]
