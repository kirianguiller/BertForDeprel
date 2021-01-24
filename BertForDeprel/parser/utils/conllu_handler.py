import conllu


path_conllu = "/home/wran/corpus/treebank/SUD/SUD_Naija/SUD_Naija-NSC/agregate_file/sud-naija_nsc_DG.conllu"

with open(path_conllu, 'r', encoding="utf-8") as f:
    sentences = conllu.parse(f.read())

sentences[-1][0]["misc"]["deprel_pred"] = "a test"
print(sentences[-1][0]["misc"])
print(sentences[-1].serialize())

# CoNLLu_pred = namedtuple(typename='CoNLLu_pred',
#                    field_names=['ID', 'FORM', 'LEMMA', 'CPOS', 'POS',
#                                 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL'],
#                    defaults=[None]*10)

# start, sentences = 0, []

# with open(path_conllu, 'r', encoding='utf-8') as f:
#     lines = [line.strip() for line in f]
#     # lines = f.readlines()
#     for i, line in enumerate(lines):
#         if not line:
#             # values = list(zip(*[l.split('\t') for l in lines[start:i]]))
#             # sentences.append(Sentence(fields, values))

#             start = i + 1

#             # print(i, values)
#         # print(i, line)

# # print(lines)
# print(start)
# print(i)
# print(lines[start:i])

# properties = {}
# tokens = []
# for line in lines[start:i]:
#     if line.startswith("#"):
#         property, value = line.split('=', 1)
#         property = property.strip("#").strip()
#         print(value)

#         properties[property] = value

#     else:
#       tokens.append(line.split("\t"))
#       print(line.split("\t"))

# print(list(zip(*tokens)))

