"""Test for word tokenizers"""
from jel.biencoder.dataset_reader import SmallJaWikiReader
from jel.biencoder.parameters import BiEncoderExperiemntParams
import pdb

def test_small_dataset_reader():
    p = BiEncoderExperiemntParams()
    config = p.opts
    reader = SmallJaWikiReader(config=config)
    train = reader._read('train')