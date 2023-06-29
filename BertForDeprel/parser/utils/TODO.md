# TODO's

* Improve "LOG: label '↑0¦↓1¦↑-2;d¦' was not found in the label2index mapping" message; this should give the conllu file, line number, and line content
* Overall grok with documentation, adding paper links, fixing typos, etc.
* Support CUR_DIR variable for the directory containing the config file
* Add calls to torch.compile and see if it speeds up
* Progress bar for batches in training epochs
* Fix out-of-filehandles error
* Generate architecture diagram
* Document how to treat training data during fine-tune. Do I just train on the new data, or do I train on the new data + the old data?
* What is adapter_config_type? Can I remove it?
* Add example using UD data
* Clarify that conf file for prediction should be the one generated in the model directory, not the one that was used during training
* Output prediction accuracy when the input file is annotated already
    - it's unclear to me that prediction actually occurred... the file just looks like a copy of the original. Seems to just be adding = signs at the end of one of the columns and #newpar indicators.
* Add a flag to output the predictions in conllu format
* Research: UDPipe 2.0-based lemmatization has a bias towards assuming that suffixation is the most common morphological process in a language. Some languages use circumfixation and I'm certain some use prefixation. Maybe we can experiment here?
* Research: Trying different adapter methods could also be interesting.
    - non-linear set to GELU
    - reduction_factor set to 2
    - plenty more configs to try...
    - PfeifferInvConfig() instead of Pfeiffer
    - UniPELTConfig() instead of Pfeiffer

* explain this warning:
```
Some weights of the model checkpoint at xlm-roberta-large were not used when initializing XLMRobertaModel: ['lm_head.dense.bias', 'lm_head.layer_norm.weight', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight', 'lm_head.bias']
- This IS expected if you are initializing XLMRobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing XLMRobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
```
