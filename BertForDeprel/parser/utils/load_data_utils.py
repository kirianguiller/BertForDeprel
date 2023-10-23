from pathlib import Path
from typing import Dict, Iterable, List

from conllup.conllup import readConlluFile, sentenceJson_T


# batch classes are for entire datasets of parses
def resolve_conllu_paths(path: Path) -> List[Path]:
    if path.is_file():
        if path.name.endswith(".conllu"):
            paths = [path]
        else:
            raise BaseException(
                "input file was not .conll neither a folder of conllu : ", path
            )
    elif path.is_dir():
        paths = list(path.glob("*.conllu"))
        if paths == []:
            raise BaseException(f"No conllu was found in path_folder=`{path}`")
    else:
        raise Exception(f"No conllu was found in path_folder=`{path}` (error 2)")
    return paths


def load_conllu_sentences(file_or_dir_path: Path) -> Iterable[sentenceJson_T]:
    sentences = load_conllu_sentences_mapping(file_or_dir_path)
    return (
        sentence for sentence_list in sentences.values() for sentence in sentence_list
    )


def load_conllu_sentences_mapping(file_or_dir_path: Path):
    """Return dict of sentences by their file path."""
    sentences: Dict[Path, List[sentenceJson_T]] = {}
    for path in resolve_conllu_paths(file_or_dir_path):
        sentences[path] = readConlluFile(str(path))
    return sentences
