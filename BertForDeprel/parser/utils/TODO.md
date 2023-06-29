# TODO's

* Improve "LOG: label '↑0¦↓1¦↑-2;d¦' was not found in the label2index mapping" message; this should give the conllu file, line number, and line content
* Overall grok with documentation, adding paper links, fixing typos, etc.
* Support CUR_DIR variable for the directory containing the config file
* Add calls to torch.compile and see if it speeds up
* Progress bar for batches in training epochs
* Fix out-of-filehandles error
* Epoch timing data should be saved in history file
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
* Try different initialization for BiAffineTranKit parameter:
    - self.pairwise_weight.data.zero_() -> try he normal or xavier uniform
* Port functionality from Trankit/UDPipe:
    - paragraph segmentation
    - sentence segmentation
    - tokenization
    - multi-word token expansion
* Add multi-word expression detection (need a data set for this)
    - or other stuff in the UD/SUD spoken corpora annotation guidelines paper from Kahane et al.
        - Could probably do data augmentations to generate disfluencies, corrections, re-wordings, un-finished sentences, co-constructions, etc. with correct tags
* Lemmatizer: try Unicode denormalization of the input, normalization of the output (when can't be normalized, use a different script) to make data less sparse
    - could also try romanize/de-romanize, for e.g. syllabaries
* Lemmatizer: try a seq2seq model
    - reported elsewhere as state-of-the-art (Trankit, Stanza, and https://github.com/jmnybl/universal-lemmatizer/ use it), but the older one here seems to out-perform it with just 10 epochs on EN-GUM, at least locally.
* grok all evaluation metrics output
    - do we want to add any more? https://trankit.readthedocs.io/en/latest/performance.html
* explain this warning:
```
Some weights of the model checkpoint at xlm-roberta-large were not used when initializing XLMRobertaModel: ['lm_head.dense.bias', 'lm_head.layer_norm.weight', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight', 'lm_head.bias']
- This IS expected if you are initializing XLMRobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing XLMRobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
```
* Try using the Chu-Liu/Edmond's available from PyPi: https://github.com/ufal/chu_liu_edmonds (just curious if it would be plug-n-play for us; not sure if we have a bottleneck there but it is C++ so it might be faster)
