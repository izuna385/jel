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
  