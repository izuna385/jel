import argparse
import sys, json
from distutils.util import strtobool

class BiEncoderExperiemntParams:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Japanese Entity linker parameters for experiment')
        parser.add_argument('-debug', action='store', default=False, type=strtobool)
        parser.add_argument('-debug_data_num', action='store', default=100, type=int)
        parser.add_argument('-biencoder_dataset_file_path', action='store', default='./data/jawiki_small_dataset/data.json', type=str)
        parser.add_argument('-title2doc_file_path', action='store', default='./data/jawiki_small_dataset/title2doc.json', type=str)

        # for training
        parser.add_argument('-max_context_window_size', action='store', default=30, type=int)
        parser.add_argument('-max_mention_size', action='store', default=15, type=int)
        parser.add_argument('-max_ent_considered_sent_num', action='store', default=10, type=int)

        parser.add_argument('-max_title_token_size', action='store', default=15, type=int)
        parser.add_argument('-max_ent_desc_token_size', action='store', default=100, type=int)

        parser.add_argument('-lr', action='store', default=1e-5, type=float)
        parser.add_argument('-num_epochs', action='store', default=2, type=int)
        parser.add_argument('-batch_size_for_train', action='store', default=96, type=int)
        parser.add_argument('-batch_size_for_eval', action='store', default=128, type=int)

        self.opts = parser.parse_args(sys.argv[1:])
        print('\n===PARAMETERS===')
        for arg in vars(self.opts):
            print(arg, getattr(self.opts, arg))
        print('===PARAMETERS END===\n')

    def get_params(self):
        return self.opts

    def dump_params(self, experiment_dir):
        parameters = vars(self.get_params())

        with open(experiment_dir + 'parameters.json', 'w') as f:
            json.dump(parameters, f, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))