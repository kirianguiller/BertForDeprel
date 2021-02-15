
# create project directory
# rm -r naija_pretrained_test # in case already existing
mkdir naija_pretrained_test/

# create conllu folder
cd naija_pretrained_test/ 
mkdir ./conllus/ 
# download train and test conllu
gdown --id 1rRhnnqz1U2vrGS0dOzqjbNjHCehEez2l  -O naija_train.conllu # naija_train.conllu
gdown --id 1gIqhBXO-phcH2pdThVFGR9eEkA2b37Dr  -O naija_test.conllu # naija_test.conllu
gdown --id 108KbxMVv0XTnUXZpS9dWoTBUkCKWRn3F  -O all_treebanks_english.conllu # all_treebanks_english
# move these conllu to the conllu folder
mv *.conllu conllus/ 

# download and move the pretrained model
gdown --id 1-569rziTZQcHr49SmGxeBsGcB_PlvPTC  -O naija_from_scratch.pt # naija_from_scratch.pt
mkdir ./models/
mv naija_from_scratch.pt ./models/

# download and move the annotation schema
gdown --id 1-20i829t0wcnt9RnlTjsPYAyz4ZiN-m- -O annotation_schema.json


# create to_predict conllu file
mkdir ./to_predict/
cp ./conllus/test.conllu ./to_predict/