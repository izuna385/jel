import argparse
import sys, json
from distutils.util import strtobool
from jel.common_config import CACHE_ROOT

class BiEncoderExperiemntParams:
    '''
    Configuration files for training biencoder.
    '''
    def __init__(self):
        parser = argparse.ArgumentParser(description='Japanese Entity linker parameters for experiment')
        parser.add_argument('-debug', action='store', default=False, type=strtobool)
        parser.add_argument('-debug_data_num', action='store', default=1000, type=int)
        parser.add_argument('-vocab_dir', action='store', default=str(CACHE_ROOT)+'/resources/vocab_dir/', type=str)
        parser.add_argument('-serialization_dir', action='store',
                            default=str(CACHE_ROOT)+'/resources/serialization_dir/chive_boe/', type=str)
        parser.add_argument('-shutil_pre_finished_experiment', action='store', default=False, type=strtobool)
        parser.add_argument('-biencoder_dataset_file_path', action='store', default='./data/jawiki_small_dataset_sudachi/data.json', type=str)
        parser.add_argument('-title2doc_file_path', action='store', default='./data/jawiki_small_dataset_sudachi/title2doc.json', type=str)

        # for training
        parser.add_argument('-max_context_window_size', action='store', default=30, type=int)
        parser.add_argument('-max_mention_size', action='store', default=15, type=int)
        parser.add_argument('-max_ent_considered_sent_num', action='store', default=10, type=int)

        parser.add_argument('-max_title_token_size', action='store', default=15, type=int)
        parser.add_argument('-max_ent_desc_token_size', action='store', default=100, type=int)

        parser.add_argument('-lr', action='store', default=5e-3, type=float)
        parser.add_argument('-num_epochs', action='store', default=10, type=int)
        parser.add_argument('-batch_size_for_train', action='store', default=20000, type=int)
        parser.add_argument('-batch_size_for_eval', action='store', default=20000, type=int)

        # bert and chive is currently available.
        parser.add_argument('-word_langs_for_training', action='store', default='chive', type=str)

        self.all_opts = parser.parse_known_args(sys.argv[1:])
        self.opts = self.all_opts[0]
        # print('\n===PARAMETERS===')
        # for arg in vars(self.opts):
        #     print(arg, getattr(self.opts, arg))
        # print('===PARAMETERS END===\n')

    def get_params(self):
        return self.opts

    def dump_params(self, experiment_dir):
        parameters = vars(self.get_params())

        with open(experiment_dir + 'parameters.json', 'w') as f:
            json.dump(parameters, f, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))