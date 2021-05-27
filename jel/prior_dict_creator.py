'''
create prior dictionary from preprocessed_jawiki or preprocessed_jawiki_sudachi.
Definition of prior: See formula (6) in https://www.aclweb.org/anthology/K19-1049.pdf
'''
from typing import List, Tuple, Dict
from glob import glob
from jel.utils.common import jopen
import json
from collections import defaultdict, Counter
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

def _m2e_collector(dataset_dir: str,
                   prior_dict_path: str,
                   debug: bool=False) -> None:
    '''
    :param dataset_dir: preprocessed dataset directory where annotations exist.
    :return: dump m2prior dict
    '''
    all_m2e = list()
    json_path_list = glob(dataset_dir+'**/*.json')
    if debug:
        json_path_list = json_path_list[:500]

    n_cores = multi.cpu_count()
    with Pool(n_cores) as pool:
        imap = pool.imap_unordered(_m2_collect_from_one_json, json_path_list)
        m2e_result = list(tqdm(imap, total=len(json_path_list)))

    for m2e_from_one_json in m2e_result:
        all_m2e += m2e_from_one_json

    m2e_dict = defaultdict(lambda: Counter())
    for (text, index) in all_m2e:
        m2e_dict[text][index] += 1

    m2prior_dict = {}
    for (m, cand_entities) in m2e_dict.items():
        all_counts_of_m2e_links = sum([count for count in cand_entities.values()])
        priors = sorted([(e, c / all_counts_of_m2e_links) for (e, c) in cand_entities.items()],
                                    key=lambda x: x[1], reverse=True)

        # TODO: Remove Disambiguation Page.
        # TODO: Resolve Redirects.
        m2prior_dict.update({m: priors})

    with open(prior_dict_path, 'w') as pdp:
        json.dump(m2prior_dict, pdp, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))


# if __name__ == '__main__':
#     dataset_dir = './data/preprocessed_jawiki_sudachi/'
#     prior_dict_path='./resources/prior_dict.json'
#     _m2e_collector(dataset_dir=dataset_dir,
#                    prior_dict_path=prior_dict_path,
#                    debug=False)
