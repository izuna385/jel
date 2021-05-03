from jel.utils.tokenizer import JapaneseBertTokenizer
import pytest
import pdb

def test_tokenize_with_anchored_text():
  anchored_txt_sample = "福岡県<a>福岡</a>市"
  tokenizer = JapaneseBertTokenizer()
  tokenized = tokenizer.tokenize(anchored_txt_sample)
  assert tokenized == ['福', '岡', '県', '<a>', '福', '岡', '</a>', '市']