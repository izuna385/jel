from glob import glob
import argparse
import os
import sys
import pdb
from typing import Dict, List
import json
import time
import random
random.seed(42)
import copy
import math
from multiprocessing import Pool
import multiprocessing as multi
from tqdm import tqdm
from jel.utils.tokenizer import SudachiTokenizer
Sudachi_Tokenizer_Class = SudachiTokenizer()

def jopen(json_filepath: str) -> Dict:
    with open(json_filepath, 'r') as f:
        j = json.load(f)

    return j

def main(dirpath_for_preprocessed_jawiki: str,
         output_small_dataset_dirpath: str,
         minimum_entity_collections: int,
         minimum_annotation_count: int):
    '''
    :param dirpath_for_preprocessed_jawiki: preprocessed files from https://github.com/izuna385/Wikia-and-Wikipedia-EL-Dataset-Creator
    Or, just download https://drive.google.com/file/d/11_SUXM5wba1fSjF7eaTFO8ISk53nEwXk/view?usp=sharing to './data/' and then unzip.

    :param output_small_dataset_dirpath:
    :return:
    '''
    if not os.path.exists(output_small_dataset_dirpath):
        os.makedirs(output_small_dataset_dirpath)

    entire_json_file_paths = glob(dirpath_for_preprocessed_jawiki+'**/*')

    # To collect various annotations, shuffle paths.
    random.shuffle(entire_json_file_paths)

    small_entitiy_collections = {}
    for json_path in entire_json_file_paths:
        j = jopen(json_path)
        annotations, doc_title2sents = j['annotations'], j['doc_title2sents']
        if len(small_entitiy_collections) > minimum_entity_collections:
            break

        for ent_title, its_desc in doc_title2sents.items():
            small_entitiy_collections[ent_title] = its_desc
            if (len(small_entitiy_collections)) % 1000 == 0:
                print('entity num:', len(small_entitiy_collections))

    print('collected entity counts:', len(small_entitiy_collections))
    annotations_whose_gold_exist_in_small_entity_collections = list()

    print('Colleting annotations...')
    start_time = time.time()
    for json_path in entire_json_file_paths:
        j = jopen(json_path)
        annotations, doc_title2sents = j['annotations'], j['doc_title2sents']

        if len(annotations_whose_gold_exist_in_small_entity_collections) > minimum_annotation_count:
            break
        for annotation in annotations:
            gold_entity = annotation['annotation_doc_entity_title']
            if gold_entity in small_entitiy_collections:
                annotations_whose_gold_exist_in_small_entity_collections.append(annotation)
            tmp_time = time.time()

            if tmp_time - start_time > 4:
                start_time = copy.copy(tmp_time)
                print('Current collected annotation:', len(annotations_whose_gold_exist_in_small_entity_collections))

    print('Collected annotation num:', len(annotations_whose_gold_exist_in_small_entity_collections))

    # dump annotations
    if not os.path.exists(output_small_dataset_dirpath):
        os.mkdir(output_small_dataset_dirpath)

    random.shuffle(annotations_whose_gold_exist_in_small_entity_collections)

    train_frac, dev_frac, test_frac = 0.7, 0.15, 0.15
    train_data_num = math.floor(len(annotations_whose_gold_exist_in_small_entity_collections) * train_frac)
    dev_data_num = math.floor(len(annotations_whose_gold_exist_in_small_entity_collections) * dev_frac)

    train, dev, test = annotations_whose_gold_exist_in_small_entity_collections[:train_data_num], \
                       annotations_whose_gold_exist_in_small_entity_collections[train_data_num: train_data_num + dev_data_num], \
                       annotations_whose_gold_exist_in_small_entity_collections[train_data_num + dev_data_num:]

    with open(output_small_dataset_dirpath + 'title2doc.json', 'w') as sdd:
        json.dump(small_entitiy_collections, sdd, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))

    with open(output_small_dataset_dirpath + 'data.json', 'w') as smd:
        json.dump({'train': train,
                   'dev': dev,
                   'test': test}, smd, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dirpath_for_preprocessed_jawiki',
        help="Path to the dirpath_for_preprocessed_jawiki file.",
        default='./data/preprocessed_jawiki_sudachi/',
        type=str
    )
    parser.add_argument(
        '--output_small_dataset_dirpath',
        help="Path to the output small dataset directory.",
        default='./data/jawiki_small_dataset_sudachi/',
        type=str
    )
    parser.add_argument(
        '--minimum_entity_collections',
        help="Minimum entity counts for creating small dataset.",
        default=10000,
        type=int
    )
    parser.add_argument(
        '--minimum_annotation_count',
        help="Minimum entity counts for creating small dataset.",
        default=50000,
        type=int
    )

    args = parser.parse_args()
    main(args.dirpath_for_preprocessed_jawiki, args.output_small_dataset_dirpath,
         args.minimum_entity_collections, args.minimum_annotation_count)