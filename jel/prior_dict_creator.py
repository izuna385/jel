'''
create prior dictionary from preprocessed_jawiki or preprocessed_jawiki_sudachi.
Definition of prior: See formula (6) in https://www.aclweb.org/anthology/K19-1049.pdf
'''
from typing import List, Tuple
from glob import glob
from jel.utils.common import jopen
import pdb
from multiprocessing import Pool
import multiprocessing as multi
from tqdm import tqdm

def _m2_collect_from_one_json(json_path: str) -> List[Tuple[str,str]]:
    annotations = jopen(json_path)['annotations']
    m2e = list()
    for annotation in annotations:
        mention, destination_of_its_mention_doc_title = annotation['mention'], annotation['annotation_doc_entity_title']
        if destination_of_its_mention_doc_title != None:
            m2e.append((mention, destination_of_its_mention_doc_title))

    return m2e

def _m2e_collector(dataset_dir: str) -> List[Tuple[str, str]]:
    '''
    :param dataset_dir: preprocessed dataset directory where annotations exist.
    :return: tuples of (mention, its_destination_title_doc ( or, after all, entity. ))
    '''
    all_m2e = list()
    json_path_list = glob(dataset_dir+'**/*.json')

    n_cores = multi.cpu_count()
    with Pool(n_cores) as pool:
        imap = pool.imap_unordered(_m2_collect_from_one_json, json_path_list)
        m2e_result = list(tqdm(imap, total=len(json_path_list)))

    for m2e_from_one_json in m2e_result:
        all_m2e += m2e_from_one_json

    return all_m2e

if __name__ == '__main__':
    dataset_dir = './data/preprocessed_jawiki_sudachi/'
    _m2e_collector(dataset_dir=dataset_dir)
