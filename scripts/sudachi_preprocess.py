from glob import glob
import argparse
import os
import sys
import pdb
from typing import Dict, List, Tuple
import json
import time
import random
random.seed(42)
import copy
import math
from multiprocessing import Pool
import multiprocessing as multi
from tqdm import tqdm
import hashlib
from jel.utils.tokenizer import SudachiTokenizer
Sudachi_Tokenizer_Class = SudachiTokenizer()

JAWIKI_PREPROCESSED_DATA_DIRPATH = './data/preprocessed_jawiki/'
JAWIKI_PREPROCESSED_DATA_SUDACHI_TOKENIZED_ADDED_DIRPATH = './data/preprocessed_jawiki_sudachi/'
MAX_CONSIDERED_SENTENCE_FOR_EACH_ENT = 10
CACHED_DIR = './sudachi_cache/'

def all_json_filepath_getter_from_preprocessed_jawiki(dirpath: str) -> List[str]:
    return glob(dirpath+'**/*')

def all_json_dirpath_getter_from_preprocessed_jawiki(dirpath: str) -> List[str]:
    return glob(dirpath+'**/')

def jopen(json_path: str) -> Tuple[Dict, Dict]:
    with open(json_path, 'r') as f:
        j = json.load(f)

    return j['annotations'], j['doc_title2sents']

def tokenize(txt: str) -> List[str]:
    return Sudachi_Tokenizer_Class.tokenize(txt=txt)

def main(json_paths_preprocessed: List[str]) -> None:
    n_cores = multi.cpu_count()
    with Pool(n_cores) as pool:
        imap = pool.imap(multiprocess_sudachi_tokenized_data_adder, json_paths_preprocessed)
        result = list(tqdm(imap, total=len(json_paths_preprocessed)))

def retrieve_cache_or_tokenize(txt: str):
    original_sentence_hash = hashlib.sha256(txt.encode("utf-8")).hexdigest()
    cache_sentence_path = os.path.join(CACHED_DIR, original_sentence_hash)
    cache_sentence_path = cache_sentence_path + '.json'
    if os.path.exists(cache_sentence_path):
        with open(cache_sentence_path, 'r') as cache:
            c = json.load(cache)
        cached_tokenized = c['cached_tokenized']

        return cached_tokenized

    else:
        tokenized = tokenize(txt)
        with open(cache_sentence_path, 'w') as cache_:
            json.dump({'cached_tokenized': tokenized}, cache_)

        return tokenized

def multiprocess_sudachi_tokenized_data_adder(json_path: str) -> int:
    annotations, doc_title2sents = jopen(json_path)

    new_annotations = list()
    for annotation in annotations:
        document_title = annotation['document_title']
        anchor_sent = annotation['anchor_sent']
        annotation_doc_entity_title = annotation['annotation_doc_entity_title']
        mention = annotation['mention']
        original_sentence = annotation['original_sentence']
        original_sentence_mention_start = annotation['original_sentence_mention_start']
        original_sentence_mention_end = annotation['original_sentence_mention_end']

        try:
            sudachi_mention = tokenize(mention)
            sudachi_anchor_sent = tokenize(anchor_sent)
            annotation.update({'sudachi_anchor_sent': sudachi_anchor_sent})
            annotation.update({'sudachi_mention': sudachi_mention})

            new_annotations.append(annotation)
        except:
            pass

    new_doc_title2sents = {}

    for ent_name, documents in doc_title2sents.items():
        documents = documents[:MAX_CONSIDERED_SENTENCE_FOR_EACH_ENT]
        title = tokenize(ent_name)
        new_sents = list()
        for sent in documents:
            try:
                tokenized = tokenize(sent)
                new_sents.append(tokenized)
            except:
                continue

        new_doc_title2sents.update({ent_name: {'sudachi_tokenized_title': title, 'sudachi_tokenized_sents': new_sents}})

    new_json_path = json_path.replace('preprocessed_jawiki', 'preprocessed_jawiki_sudachi')
    with open(new_json_path, 'w') as njp:
        json.dump({'annotations': new_annotations, 'doc_title2sents':  new_doc_title2sents}, njp,
                  ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))

    return 1

if __name__ == '__main__':
    json_paths_preprocessed = all_json_filepath_getter_from_preprocessed_jawiki(dirpath=JAWIKI_PREPROCESSED_DATA_DIRPATH)

    # dirpath create for sudachi preprocessing
    dirpaths_preprocessed = all_json_dirpath_getter_from_preprocessed_jawiki(dirpath=JAWIKI_PREPROCESSED_DATA_DIRPATH)
    new_dirpaths_for_sudachi = [dirpath.replace('preprocessed_jawiki', 'preprocessed_jawiki_sudachi') for dirpath in dirpaths_preprocessed]
    for dirpath in new_dirpaths_for_sudachi:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    main(json_paths_preprocessed)