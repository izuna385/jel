from jel.utils.common import return_ner_span
import pytest
import pdb

def test_return_ner_span():
  text_sample = "夏目漱石と本郷三丁目"
  spans = return_ner_span(text_sample)
  assert spans == [{'text': '夏目漱石', 'span': (0, 4)}, {'text': '本郷三丁目', 'span': (5, 10)}]