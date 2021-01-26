
# Tutorial End-to-End

Google colab showing how to use this parser are available here : 
- training from scratch : https://colab.research.google.com/drive/1J50pOlBnY-sCliBTinF-9soK6LZRZndn

## Prepare Dataset
Create a folder with the following structure :
```
|- [NAME_FOLDER]/
|   |- conllus/
|       | - <train.conllu>
|       | - <test.conllu>
```
where `<train.conllu>` and `<test.conllu>` are respectively the train and test datasets. They can have the name you want as you will have to indicate the path to this file in the running script.


## Compute the annotation schema
The annotation schema is the set of dependency relation (deprel) and part-of-speeches (upos/pos) that will be required for the model to know the size of the classifiers layers. Once a model is trained on a given annotation schema, it is **required** to use the same annotation schema for inference and fine-tuning.

For computing this annotation schema, run the script `<root_repo>/BertForDeprel/preprocessing/1_compute_annotation_schema.py` with the parameter `-i --input_folder` linking to the train folder and the `-o --output_path` parameter linking to the location of the annotation schema to be writter.

After preprocessing the annotation schema, the structure of the project folder should be:
```
|- [NAME_FOLDER]/
|   |- conllus/
|       | - <train.conllu>
|       | - <test.conllu>
|   |- <annotation_schema.json>
```

## Training models

### From scratch

To train a model from scratch, you can, from the `BertForDeprel/` folder, run the following command :

```
python run.py train --folder ../test/test_folder/ --model mode_name.pt --bert_type bert-base-multilingual-cased --ftrain ../test/test_folder/conllus/train.conll
```

where `--folder` indicate the path to the project folder, `--model` the name of the model to be trained, `--ftrain` the path to the train conll. If the optionnal parameter `--ftest` is passed, the corresponding file will be used for test. Otherwise, the model will automatically split the train dataset in `--split_ratio` with a random seed of `--random_seed`.



