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

ENCODER_DIRPATH = './resources/encoders/'
MODEL_TAR_GZ_DIRPATH = './resources/'
RESOURCES_DIRPATH = './resources/'

# for collect_entity_data.py
ENTITY_DATA_PATH = './data/preprocessed_jawiki_sudachi/**/*'
ENTITY_VEC_DIR_PATH = './resources/entity_name2vec/'
PRIOR_DICT_PATH = './resources/prior_dict.json'