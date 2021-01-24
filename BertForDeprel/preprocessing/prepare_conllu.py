import argparse
import os
import conllu
from transformers import BertTokenizer, CamembertTokenizer, AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser model.'
    )

    parser.add_argument('--folder', '-f', required=True,
                               help='path to project folder')
    parser.add_argument('--bert_type', '-b', default='bert',
                               help='bert type to use (bert/camembert)')
    
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                 const=sum, default=max,
    #                 help='sum the integers (default: find the max)')

    args = parser.parse_args()


    if args.bert_type == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.bert_type == "camembert":
        tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    elif args.bert_type == "mbert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    else :
        tokenizer = AutoTokenizer.from_pretrained(args.bert_type)

    
    path_teacher_folder = os.path.join(args.folder, "teacher")
    path_student_folder = os.path.join(args.folder, "student")

    with open(os.path.join(path_teacher_folder, os.listdir(path_teacher_folder)[0])) as infile:
        parsed_conllu = conllu.parse(infile.read())

    for sequence in parsed_conllu:
        len_sequence = 2        #artificially count CLS and SEP token
        for token in sequence:
            len_sequence += len(tokenizer.encode(token['form'], add_special_tokens=False))
          
        if len_sequence > 254:
            print("size", len_sequence)
            print("sequence", sequence)
