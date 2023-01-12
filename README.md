Source code for the paper:

Finetuned Language Models are Zero-Shot Learners, Wei et. al., 2021.

## Install instructions

* Install t5, seqio
* `ln -s ./flan ~/prompt-tuning/flan`
* (Possibly change the vocabulary path in flan/tasks.py, eg to: gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model)
