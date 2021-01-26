
# Tutorial End-to-End


## Prepare Dataset
Create a folder with the following structure :
```
|- [NAME_FOLDER]/
|   |- train/
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
|   |- train/
|       | - <train.conllu>
|       | - <test.conllu>
|   |- <annotation_schema.json>
```

