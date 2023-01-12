"""Check FLAN tasks."""
import argparse
import json
import os
import traceback

from flan import tasks  # pylint: disable=unused-import
import numpy as np
import seqio


TASKS = [x + '_type_0' for x in '''
aeslc
ag_news_subset
anli_r1
anli_r2
anli_r3
arc_challenge
arc_easy
bool_q
cb
cnn_dailymail
cola
common_gen
copa
coqa
cosmos_qa
dart
definite_pronoun_resolution
drop
e2e_nlg
fix_punct
gigaword
glue_mrpc
glue_qqp
hellaswag
imdb_reviews
math_dataset
mnli_matched
mnli_mismatched
multirc
natural_questions
openbookqa
opinion_abstracts_idebate
opinion_abstracts_rotten_tomatoes
para_crawl_enes
paws_wiki
piqa
qnli
quac
record
rte
samsum
sentiment140
snli
squad_v1
squad_v2
sst2
story_cloze
stsb
trec
trivia_qa
true_case
web_nlg_en
wic
wiki_lingua_english_en
winogrande
wmt14_enfr
wmt16_translate_csen
wmt16_translate_deen
wmt16_translate_fien
wmt16_translate_roen
wmt16_translate_ruen
wmt16_translate_tren
wnli
word_segment
wsc
xsum
yelp_polarity_reviews
'''.strip().split() if not x.startswith('wmt') and x != 'para_crawl_enes']


def is_good_task(name):
  return name in TASKS


def to_json_value(value):
  """Convert the value to JSON-serializable value."""
  if isinstance(value, np.ndarray):
    return to_json_value(value.tolist())
  if isinstance(value, list):
    return [to_json_value(x) for x in value]
  if isinstance(value, bytes):
    return value.decode()
  if hasattr(value, "item"):
    return value.item()
  return value


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('outdir')
  args = parser.parse_args()
  os.makedirs(args.outdir, exist_ok=True)
  registered_tasks = seqio.TaskRegistry.names()
  with open(os.path.join(args.outdir, "all-tasks.txt"), "w") as fout:
    for name in sorted(registered_tasks):
      print(name, file=fout)
  registered_tasks = [name for name in registered_tasks if is_good_task(name)]
  for name in sorted(registered_tasks):
    print(f"Processing task {name}")
    outjsonl = os.path.join(args.outdir, name + ".jsonl")
    outtext = os.path.join(args.outdir, name + ".txt")
    with open(outjsonl, "w") as fjsonl, open(outtext, 'w') as ftext:
      try:
        task = seqio.get_mixture_or_task(name)
        dataset = task.get_dataset(
            sequence_length={
                "inputs": 512,
                "targets": 512,
            },
            split="validation",
            shuffle=False,
        )
        for _, ex in zip(range(100), dataset.as_numpy_iterator()):
          ex_dict = {key: to_json_value(value) for (key, value) in ex.items()}
          print(json.dumps(ex_dict), file=fjsonl)
          print('=' * 60, file=ftext)
          print(repr(ex_dict['inputs_pretokenized']), file=ftext)
          print(repr(ex_dict['targets_pretokenized']), file=ftext)
      except KeyboardInterrupt as e:
        raise e
      except:
        print(f'ERROR in task {name}!')
        traceback.print_exc()


if __name__ == "__main__":
  main()
