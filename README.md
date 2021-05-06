<p align="center"><img width="20%" src="docs/jel-logo.png"></p>

# jel: Japanese Entity Linker
* Lightweight linker (with sudachi) and transformer-based linker are suppported.

## Test
`$ python pytest`

## Development
*  `export PYTHONPATH="/Users/your_name/Desktop/github/jel/jel:${PYTHONPATH}"`

## memo
* @ batch_size_for_train with bert, 1,07iter / s. @batch_size_for_eval with 128, 1.61 iter/s

## setup for sudachi experiment.
```
$ python3 ./scripts/sudachi_preprocess.py
$ python3 ./scripts/small_dataset_creator.py
$ python3 ./scripts/biencoder_training_check.py
```

## Notes
* faiss==1.5.3 from pip causes error _swigfaiss. 

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