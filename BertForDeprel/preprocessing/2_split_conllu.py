# script for splitting conllus across various k-fold train/dev/test


import argparse
import os
import conllu
from sklearn.model_selection import train_test_split 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser model.'
    )

    parser.add_argument('--file', '-f', required=True,
                               help='path to file to split')
    parser.add_argument('--n_echantillon', default=0, type=int,
                                help='number of sequences to use for parsing (used in the experience 1/10/100/1000...')                               
    # parser.add_argument('--bert_type', '-b', default='bert',
    #                            help='bert type to use (bert/camembert)')
    
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                 const=sum, default=max,
    #                 help='sum the integers (default: find the max)')

    args = parser.parse_args()


    # if args.bert_type == "bert":
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # elif args.bert_type == "camembert":
    #     tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    # elif args.bert_type == "mbert":
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

    
    # path_teacher_folder = os.path.join(args.folder, "teacher")

    for seed in range(1,6):
        path_train = "/".join(args.file.split("/")[:-1]) + '/{}_seed_{}.conllu'.format("train", seed)
        path_test = "/".join(args.file.split("/")[:-1]) + '/{}_seed_{}.conllu'.format("test", seed)
        with open(args.file, 'r', encoding='utf-8') as infile:
            parsed_conllu = conllu.parse(infile.read())
        print(len(parsed_conllu))
        print(type(parsed_conllu))
        # for sequence in parsed_conllu:

        if args.n_echantillon:
            parsed_conllu = parsed_conllu[:args.n_echantillon]
        parsed_conllu_train, parsed_conllu_test = train_test_split(parsed_conllu, test_size=0.20, random_state=seed )
        print(len(parsed_conllu_train), len(parsed_conllu_test))


        with open(path_train, 'w') as f:
            f.writelines([sequence.serialize() for sequence in parsed_conllu_train])

        with open(path_test, 'w') as f:
            f.writelines([sequence.serialize() for sequence in parsed_conllu_test])
