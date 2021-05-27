'''
KnowledgeBase Class
'''
import faiss
import numpy as np
import pickle
from glob import glob
from jel.common_config import ENTITY_DATA_PATH, ENTITY_VEC_DIR_PATH, RESOURCES_DIRPATH
from tqdm import tqdm
import os
import pdb

class TitleIndexerWithFaiss:
    def __init__(self, kbemb_dim=300,
                 search_method_for_faiss='indexflatip',
                 how_many_top_hits_preserved=20):
        self.kbemb_dim = kbemb_dim
        self.entity_idx2emb, self.entity_id2title = self._entity2vec_loader()
        self.entity_title2id = {}
        for idx, title in self.entity_id2title.items():
            self.entity_title2id.update({''.join(title): idx})
        self.entity_num = len(self.entity_idx2emb)
        self.search_method_for_faiss = search_method_for_faiss
        self._indexed_faiss_loader()
        self.KBmatrix, self.kb_idx2entity_idx = self._KBmatrixloader()
        self._indexed_faiss_KBemb_adder(KBmatrix=self.KBmatrix)

        self.how_many_top_hits_preserved = how_many_top_hits_preserved

    def _entity2vec_loader(self):
        if os.path.exists(RESOURCES_DIRPATH + 'entity_id2vec.pkl') and \
                os.path.exists(RESOURCES_DIRPATH + 'entity_id2name.pkl'):
            with open(RESOURCES_DIRPATH + 'entity_id2vec.pkl', 'rb') as f:
                entity_idx2emb = pickle.load(f)
            with open(RESOURCES_DIRPATH + 'entity_id2name.pkl', 'rb') as g:
                entity_id2name = pickle.load(g)

            return entity_idx2emb, entity_id2name

        pickles = glob(ENTITY_VEC_DIR_PATH+'*.pkl')
        entity_idx2emb, entity_id2name = {}, {}
        for pkl_path in tqdm(pickles):
            with open(pkl_path, 'rb') as f:
                for ent in pickle.load(f):
                    title = ent['entity_name']
                    if 'contextualized_entity' in ent['vec']:
                        vec = ent['vec']['contextualized_entity']
                    else:
                        vec = ent['vec']
                    idx = len(entity_idx2emb)
                    entity_idx2emb.update({idx: vec})
                    entity_id2name.update({idx: title})

        if not os.path.exists(RESOURCES_DIRPATH):
            os.mkdir(RESOURCES_DIRPATH)

        with open(RESOURCES_DIRPATH + 'entity_id2vec.pkl', 'wb') as f:
            pickle.dump(entity_idx2emb, f)
        with open(RESOURCES_DIRPATH + 'entity_id2name.pkl', 'wb') as g:
            pickle.dump(entity_id2name, g)

        return entity_idx2emb, entity_id2name

    def _KBmatrixloader(self):
        KBemb = np.random.randn(self.entity_num, self.kbemb_dim).astype('float32')
        kb_idx2mention_idx = {}
        for idx, (mention_idx, emb) in enumerate(self.entity_idx2emb.items()):
            KBemb[idx] = emb
            kb_idx2mention_idx.update({idx: mention_idx})

        return KBemb, kb_idx2mention_idx

    def _indexed_faiss_loader(self):
        if self.search_method_for_faiss == 'indexflatl2':  # L2
            self.indexed_faiss = faiss.IndexFlatL2(self.kbemb_dim)
        elif self.search_method_for_faiss == 'indexflatip':  #
            self.indexed_faiss = faiss.IndexFlatIP(self.kbemb_dim)
        elif self.search_method_for_faiss == 'cossim':  # innerdot * Beforehand-Normalization must be done.
            self.indexed_faiss = faiss.IndexFlatIP(self.kbemb_dim)

    def _indexed_faiss_KBemb_adder(self, KBmatrix):
        if self.search_method_for_faiss == 'cossim':
            KBemb_normalized_for_cossimonly = np.random.randn(self.entity_num, self.kbemb_dim).astype('float32')
            for idx, emb in enumerate(KBmatrix):
                if np.linalg.norm(emb, ord=2, axis=0) != 0:
                    KBemb_normalized_for_cossimonly[idx] = emb / np.linalg.norm(emb, ord=2, axis=0)
            self.indexed_faiss.add(KBemb_normalized_for_cossimonly)
        else:
            self.indexed_faiss.add(KBmatrix)

    def _indexed_faiss_returner(self):
        return self.indexed_faiss

    def search_with_emb(self, emb):
        scores, faiss_search_candidate_result_kb_idxs = self.indexed_faiss.search(
            np.array([emb]).astype('float32'),
            self.how_many_top_hits_preserved)
        top_titles, scores_from_dot = [], []

        for kb_idx, score in zip(faiss_search_candidate_result_kb_idxs[0], scores[0]):
            entity_idx = self.kb_idx2entity_idx[kb_idx]
            candidate_title = ''.join(self.entity_id2title[entity_idx])
            top_titles.append(candidate_title)
            scores_from_dot.append(score)

        return top_titles, scores_from_dot

    def title2entity_vec(self, title: str):
        if title in self.entity_title2id:
            return np.array(self.entity_idx2emb[self.entity_title2id[title]])
        else:
            return np.random.randn(300,)

if __name__ == '__main__':
    kb = TitleIndexerWithFaiss()