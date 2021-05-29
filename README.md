<p align="center"><img width="20%" src="docs/jel-logo.png"></p>

# jel: Japanese Entity Linker
* jel - Japanese Entity Linker - is Bi-encoder based entity linker for japanese.

# Usage
* Currently, `link` and `question` methods are supported.

## `el.link`
* This returnes named entity and its candidate ones from Wikipedia titles.
```python
from jel import EntityLinker
el = EntityLinker()

el.link('今日は東京都のマックにアップルを買いに行き、スティーブジョブスとドナルドに会い、堀田区に引っ越した。')
>> [
    {
        "text": "東京都",
        "label": "GPE",
        "span": [
            3,
            6
        ],
        "predicted_normalized_entities": [
            [
                "東京都庁",
                0.1084
            ],
            [
                "東京",
                0.0633
            ],
            [
                "国家地方警察東京都本部",
                0.0604
            ],
            [
                "東京都",
                0.0598
            ],
            ...
        ]
    },
    {
        "text": "アップル",
        "label": "ORG",
        "span": [
            11,
            15
        ],
        "predicted_normalized_entities": [
            [
                "アップル",
                0.2986
            ],
            [
                "アップル インコーポレイテッド",
                0.1792
            ],
            …
        ]
    }
```

## `el.question`
* This returnes candidate entity for any question from Wikipedia titles.
```python
>>> linker.question('日本の総理大臣は？')
[('菅内閣', 0.05791765857101555), ('枢密院', 0.05592481946602986), ('党', 0.05430194711042564), ('総選挙', 0.052795400668513175)]
```

## Setup
`pip install jel`

## Test
`$ python pytest`

## Notes
* faiss==1.5.3 from pip causes error _swigfaiss. 
* To solve this, see [this issue](https://github.com/facebookresearch/faiss/issues/821#issuecomment-573531694).

## LICENSE
Apache 2.0 License.

## CITATION
```
@INPROCEEDINGS{manabe2019chive,
    author    = {真鍋陽俊, 岡照晃, 海川祥毅, 髙岡一馬, 内田佳孝, 浅原正幸},
    title     = {複数粒度の分割結果に基づく日本語単語分散表現},
    booktitle = "言語処理学会第25回年次大会(NLP2019)",
    year      = "2019",
    pages     = "NLP2019-P8-5",
    publisher = "言語処理学会",
}
```