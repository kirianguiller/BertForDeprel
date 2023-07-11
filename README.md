
# Tutorial End-to-End

Google colab showing how to use this parser are available here :
- naija spoken training from pre-trained english model : [link](https://colab.research.google.com/drive/1QmM73BkeoUqi3LSeeEyh79zB2oVnf-qj?usp=sharing)
- training from scratch on naija spoken : [link](https://colab.research.google.com/drive/1j9jrxBnsRsI0d93uN3r9Kx--KumYSh86?usp=sharing)
- training from scratch on written english : [link](https://colab.research.google.com/drive/1UngKLyqRZk7vXawWnYzJtrjrNisPnhgK?usp=sharing)
- mock colab for testing if everything is fine : [link](https://colab.research.google.com/drive/1J50pOlBnY-sCliBTinF-9soK6LZRZndn?usp=sharing)

## Installation
On linux
```bash
git clone https://github.com/kirianguiller/BertForDeprel
cd BertForDeprel
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## How to run
### Train a model
Either provide the path to a model json config :
```bash
python /home/BertForDeprel/BertForDeprel/run.py train --conf /home/models/template.config.json   --ftrain /home/parsing_project/conllus/train.conllu
```

or just give a `--model_folder_path` and a `--model_name` parameter (default params will be loaded if no config or no CLI parameters are provided)
```bash
python /home/BertForDeprel/BertForDeprel/run.py train --model_folder_path /home/models/ --model_name my_parser   --ftrain /home/parsing_project/conllus/train.conllu
```

PS : here an example of a valid config.json
```json
{
    "model_folder_path": "/home/user1/models/",
    "max_epoch": 150,
    "patience": 30,
    "batch_size": 16,
    "maxlen": 512,
    "embedding_type": "xlm-roberta-large",
    "adapter_config_type": ""
}
```
### Predicting on raw conllus
For predicting, you need to provide the `--conf` parameter, which is the path to the xxx.config.json file. You also need to provide the `--inpath` parameter, which is the path to a single conllu file or a folder containing multiple conllu. The output folder parameter `--outpath` (or `-o`) is optional.
```bash
python /home/BertForDeprel/BertForDeprel/run.py train --conf /home/models/my_parser.config.json   --inpath /home/parsing_project/to_predict/ --outpath /home/parsing_project/predicted/
```

## Command line parameters

### shared

* `--conf` `-c` : path to config json file (for training, it's optional if both `--model_folder_path` and `model_name` are provided)
* `--batch_size`: numbers of sample per batches (high incidence on total speed)
* `--num_workers`: numbers of workers for preparing dataset (low incidence on total speed)
* `--seed` `-s` : random seed (default = 42)

The directory to store and load pretrained models is set via the environment variable `TORCH_HOME`.

### train

* `--model_folder_path` `-f` path to parent folder of the model : optional if `--conf` is already provided
* `--embedding_type`  `-e` : type of embedding (default : `xlm-roberta-large`)
* `--max_epoch` : maximum number of epochs (early stopping can shorten this number)
* `--patience` : number of epochs without improve required to stop the training (early stopping)
* `--ftrain` : path to train file or folder (files need .conllu extension)
* `--ftest` : path to train file or folder (files need .conllu extension) (not required. If not provided, see `--split_ratio` )
* `--split_ratio` : Ratio for splitting ftrain dataset in train and test dataset (default : 0.8)
* `--path_annotation_schema`: path to an annotation schema (json format)
* `--path_folder_compute_annotation_schema` provide a path to a folder containing various conllu, so the annotation schema is computed on these conllus before starting the training on --ftrain
* `--conf_pretrain` : path to pretrain model config, used for finetuning a pretrained BertForDeprel model
* `--overwrite_pretrain_classifiers`: erase pretraines classifier heads and recompute annotation schema

### predict

* `--inpath` `-i` : path to the file or the folder containing the files to predict
* `--outpath` `-o` : path to the folder that will contain the predicted files
* `--suffix` : optional (default = "") , suffix that will be added to the name of the predicted files (before the file extension)
* `--overwrite` : whether or not to overwrite outputted predicted conllu if already existing
* `--write_preds_in_misc` : whether or not to write prediction in the conllu MISC column instead than in the corresponding column for upos deprel and head


## Prepare Dataset
You will need some conllus for training the model and doing inferences.

### data for training
For training, you have the choice between :
- providing a single conllu file (`--ftrain` cli parameter) with all your training and testing sentences (train_test split ratio is 0.8 by default, but you can set it with `--split_ratio` parameter)
- providing a train conllu file (`--ftrain`) and a test conllu file (`--ftest`)
- providing a train folder containing the .conllu files (--ftest can also be provided, as a file or a folder too)

### data for inferences
For inference, you have to provide an input file or folder (`--inpath` or `-i`). The model will infere parse trees for all sentences of all conllus, and these outputted conllus will be written in the output folder (`--outpath` or `-o`)

### annotation schema
For people who want to use the parser for language transfer (training on lang A, then fine tuning on lang B), it is important to provide a `--path_folder_compute_annotation_schema` with a folder that contains both gold conllu from lang A and B so you can precompute the annotation schema (set of deprels, uposs, feats, lemma scripts, etc) before the pretraining. It is **required** to use the same annotation schema for training, inference and fine-tuning.

```

```
### Folder hierarchy example
Here is a folder structure example of how I am storing the different train/test/to_predicts/results conllus
```
|- [NAME_FOLDER]/
|   |- conllus/
|       | - <train.langA.conllu>
|       | - <test.langA.conllu>
|       | - <train.langB.conllu>
|       | - <test.langB.conllu>
|   |- to_predict/
|       | - <raw1.langB.conllu>
|       | - <raw2.langB.conllu>
|       | - <raw3.langB.conllu>
|   |- predicted/
```
where `<train.conllu>` and `<test.conllu>` are respectively the train and test datasets. They can have the name you want as you will have to indicate the path to this file in the running script.



## Finetuning a previously trained BertForDeprel model
WARNING : when training from a pretrained model, be sure to use the same annotation_schema.json for fine-tuning that the one that was used for pretraining. It would break the training otherwise.

To fine-tune a pre-trained model, need to follow the same step as for training a new model, but need to also provide the path to the config file of the previously trained model with `--conf_pretrained`
```bash
python /home/BertForDeprel/BertForDeprel/run.py train --model_folder_path /home/models/ --model_name my_parser  --ftrain /home/parsing_project/conllus/train.conllu  --conf_pretrained /home/models/pretrained_model.config.json
```


### GPU/CPU training
#### Run on a single GPU
For running the training on a single GPU of id 0, add the parameter `--gpu_ids 0`. Respectively, for running on one single gpu of id 3, add the parameter ``--gpu_ids 3`

#### Run on multiples GPUs
For running the training on multiple GPU of ids 0 and 1, add the parameter `--gpu_ids 0,1`

#### Run on all available GPUs
For running the training on all available GPUs, add the parameter `--gpu_ids "-2"`

#### Run on CPU
For training on CPU only, add the parameter `--gpu_ids "-1"`

## Pretrained Models
You can find [on this Gdrive repo](https://drive.google.com/drive/folders/1lVhG00JWBxrisDRytLYH3M1uZG1ZHXol?usp=sharing) all the pretrained models, google colab script for training and publicly available treebanks (.conllu files).

Among others, here are the most important pretrained models :
- [English model trained from scratch on written english](https://drive.google.com/drive/folders/1-UB0WNG8Drt_oXC7wfHlK5goMH6CC4IM?usp=sharing)
- [Naija model trained from scratch on spoken naija]("TODO")
- [Naija model fine-tuned on spoken naija from model pretrained on written english]("TODO")

## Major TODOs
- [x] Add feats and glose prediction
- [x] Add lemma
- [ ] Add confidence threshold prediction (model outputting nothing when the confidence is below a certain value)
- [ ] Add possibility of returning the confidence of the predictions (inside miscs)
