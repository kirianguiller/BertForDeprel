folder="test_folder"
ftrain="${folder}/conllus/train.conllu"

path_run="../BertForDeprel/run.py"

seed=1
bert_type="bert-base-cased"
model_name="test_model.pt"
cmd="python ${path_run} train --folder $folder --ftrain ${ftrain}  --punct --model $model_name --batch_size 8 \
--bert_type ${bert_type} --compute_fields --seed $seed ";
#--ftrain $ftrain --ftest $ftest --fpretrain $path_pretrain --compute_fields --seed $seed ";


echo -e "\nhere the command \n$cmd\n" ;
$cmd ;