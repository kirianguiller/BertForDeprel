# TODO's

Next: make sure a train/test API of any kind exists, then write a basic sanity check that sets the random seed and makes sure the results come back the same. Make sure to test speed as well. Then develop in peace 😌.

Expected characteristics:

time p3 BertForDeprel/run.py train --conf En-GUM/config.json --ftrain ~/Downloads/ud-treebanks-v2.12/UD_English-GUM/en_gum-ud-train.conllu --ftest ~/Downloads/ud-treebanks-v2.12/UD_English-GUM/en_gum-ud-test.conllu | tee En-GUM/train_out.txt

On my MP1:
0 should trake 90ish to evaluate
1 should take 346.49 to train (25 sent/sec), 60ish to eval (19 sent/sec)

time p3 BertForDeprel/run.py predict --overwrite --conf En-GUM-basic/model/config.json --inpath ~/Downloads/ud-treebanks-v2.12/UD_English-GUM/en_gum-ud-test.conllu --outpath En-GUM-basic/ | tee En-GUM-basic/predict_out.txt

On my MP1:
Should take 41.31ish (27sent/sec)

---

Address 2 "TODO: Next" refactoring items in the code.

Next: Output prediction accuracy when the input file is annotated already. Verify that performance matches expectations.
Then: build Flask server - prediction is currently hardcoded to read from and write to conllu files - need to load model in an init method and use it for subsequent requests

Update readme with new build info: poetry, poe commands, pytest, verification, pre-commit, etc.

-   Try XLM-RoBERTa-XL, which is 2 years newer than xlm-roberta
-   train_adapter is always called! This might be activating some dropouts or similar, which would be detrimental to performance.
-   Try CANINE embeddings, which would probably remove many OOV's
-   Support CUR_DIR variable for the directory containing the config file
-   Progress bar for batches in training epochs
-   Fix out-of-filehandles error
-   Epoch timing data should be saved in history file
-   What is adapter_config_type? Can I remove it?
-   Add example using UD data
-   Clarify that conf file for prediction should be the one generated in the model directory, not the one that was used during training
-   How to resume training on one paused with CMD-C?
-   Run arg to turn on allow_copy in lemma prediction
-   Clarify how to train with partially annotated data
    -   only marked morphology, only marked lemmas, etc.
-   Add a flag to output the predictions in conllu format
-   Research: UDPipe 2.0-based lemmatization has a bias towards assuming that suffixation is the most common morphological process in a language. Some languages use circumfixation and I'm certain some use prefixation. Maybe we can experiment here?
-   assert best_tree is not None -> do something better than throwing an error here
-   Why do we have an adapter on the LLM layer but then also do a down-projection in PosAndDeprelParserHead? Doesn't the adapter already do a down-projection?
-   Port functionality from Trankit/UDPipe:
    -   paragraph segmentation
    -   sentence segmentation
    -   tokenization
        -   The input tokens _must_ be in Roberta's token dictionary, otherwise we hit `Exception: The token 22 of sentence train-s1679 is not present in the tokenizer vocabulary, resulting in a 0-length token_ids vector`. Adding our own tokenization would fix this.
            -   Shouldn't we also support OOV tokens?
    -   multi-word token expansion
-   Add multi-word expression detection (need a data set for this)
    -   or other stuff in the UD/SUD spoken corpora annotation guidelines paper from Kahane et al.
        -   Could probably do data augmentations to generate disfluencies, corrections, re-wordings, un-finished sentences, co-constructions, etc. with correct tags
-   Lemmatizer: try Unicode denormalization of the input, normalization of the output (when can't be normalized, use a different script) to make data less sparse
    -   could also try romanize/de-romanize, for e.g. syllabaries
-   Lemmatizer: try a seq2seq model
    -   reported elsewhere as state-of-the-art (Trankit, Stanza, and https://github.com/jmnybl/universal-lemmatizer/ use it), but the older one here seems to out-perform it with just 10 epochs on EN-GUM, at least locally.
-   grok all evaluation metrics output
    -   do we want to add any more? https://trankit.readthedocs.io/en/latest/performance.html
-   Try using the Chu-Liu/Edmond's available from PyPi: https://github.com/ufal/chu_liu_edmonds (just curious if it would be plug-n-play for us; not sure if we have a bottleneck there but it is C++ so it might be faster)

## Done/Not Doing

-   Add calls to torch.compile and see if it speeds up
    -   Not supported in python 3.11+ yet
-   Try to_bettertransformer on the llm model
    -   nah, once you do `self.llm_layer = self.llm_layer.to_bettertransformer()`, you can't add adapters anymore (`AttributeError: 'XLMRobertaModel' object has no attribute 'add_adapter'`)
-   explain this warning:

```
Some weights of the model checkpoint at xlm-roberta-large were not used when initializing XLMRobertaModel: ['lm_head.dense.bias', 'lm_head.layer_norm.weight', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight', 'lm_head.bias']
- This IS expected if you are initializing XLMRobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing XLMRobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
```

This is expected for our case, since we are using the model on a different task. The original model has weights for performing a specific task, and we only use the parts of the model that are considered task-agnostic.

-   Research: Trying different adapter methods could also be interesting.
    -   reduction_factor set to 2
        -   not worth it
    -   gating where different head gate values are not tied together
        -   only matters for UniPelt or other combined adapters, which are not worth it
    -   PfeifferInvConfig() instead of Pfeiffer
        -   this is used for isolating task from language knowledge for zero-shot transfer learning, so it's not relevant for us
-   Research: Try using LayerNorm instead of dropout in BiAffine layer
-   Research: Try using GeLU instead of ReLU in BiAffine layer
-   Generate architecture diagram

    -   generation done with torchviz, but it's hard to understand; has every single computation. Would rather have something more summarized for understanding, which really means doing it by hand.

-   Document how to treat training data during fine-tune. Do I just train on the new data, or do I train on the new data + the old data?
    -   This will be something to tune; training on just specific data gives its features more weight. Will have to test it.
