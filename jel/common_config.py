import os
from pathlib import Path

CACHE_ROOT = Path(os.getenv("JEL_CACHE", str(Path.home() / ".jel")))

# For bi-encoder
MENTION_ANCHORS = ['<a>', '</a>']
MENTION_START_ANCHOR = MENTION_ANCHORS[0]
MENTION_END_ANCHOR = MENTION_ANCHORS[1]
MENTION_START_BERT_TOKEN = '[unused1]'
MENTION_END_BERT_TOKEN = '[unused2]'
CANONICAL_AND_DEF_BERT_CONNECT_TOKEN = '[unused3]'
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
MENTION_ANCHORS_REGEX = r'<a>|</a>'

ENCODER_DIRPATH = str(CACHE_ROOT)+'/resources/encoders/'
MODEL_TAR_GZ_DIRPATH = str(CACHE_ROOT)+'/resources/'
RESOURCES_DIRPATH = str(CACHE_ROOT)+'/resources/'

# for collect_entity_data.py
ENTITY_DATA_PATH = str(CACHE_ROOT)+'/data/preprocessed_jawiki_sudachi/**/*'
ENTITY_VEC_DIR_PATH = str(CACHE_ROOT)+'/resources/entity_name2vec/'
PRIOR_DICT_PATH = str(CACHE_ROOT)+'/resources/prior_dict.json'

RESOURCES_GOOGLE_DRIVE_ID = '1zEqZaqNbOw8cXoon7MoPdX0kGaHx0_3K'