
# Tutorial End-to-End

Google colab showing how to use this parser are available here : 
- naija spoken training from pre-trained english model : [link](https://colab.research.google.com/drive/1QmM73BkeoUqi3LSeeEyh79zB2oVnf-qj?usp=sharing) 
- training from scratch on naija spoken : [link](https://colab.research.google.com/drive/1j9jrxBnsRsI0d93uN3r9Kx--KumYSh86?usp=sharing)
- training from scratch on written english : [link](https://colab.research.google.com/drive/1UngKLyqRZk7vXawWnYzJtrjrNisPnhgK?usp=sharing)
- mock colab for testing if everything is fine : [link](https://colab.research.google.com/drive/1J50pOlBnY-sCliBTinF-9soK6LZRZndn?usp=sharing)

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




## Major TODOs
- [x] Implement the model.
- [x] Train a model from scratch on naija
- [x] Fine-tune a model on naija pretrain from scratch on english
- [ ] Enable process based distributed training. Similar to (https://github.com/fastai/imagenet-fast/).
- [ ] Implementing mixed precision (fp16) for faster training (see this [link from pytorch doc](https://pytorch.org/docs/stable/amp.html))
- [ ] Model optimization (model export, model pruning etc.)

