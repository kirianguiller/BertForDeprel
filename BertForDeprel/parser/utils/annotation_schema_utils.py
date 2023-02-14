import conllu
from lemma_script_utils import gen_lemma_rule, gen_lemma_script_from_conll_token


def create_lemma_script_list(*paths):
    lemma_scripts = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as infile:
            parsed_conllu = conllu.parse(infile.read())

        for sequence in parsed_conllu:
            for token in sequence:
                lemma_script = gen_lemma_script_from_conll_token(token)
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
    annotation_schema["deprels"] = sorted(list(set(deprels)))
    annotation_schema["uposs"] = sorted(upos)
    annotation_schema["lemma_script"] = lemma_scripts
    print(annotation_schema)
    return annotation_schema
