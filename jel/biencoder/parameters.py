import argparse
import sys, json
from distutils.util import strtobool

class BiEncoderParams:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Japanese Entity linker parameters')
        parser.add_argument('-debug', action='store', default=False, type=strtobool)

        # for training
        parser.add_argument('-batch_size_for_train', action='store', default=64, type=int)

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