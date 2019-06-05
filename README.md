# Stance Detection


## Dataset

We use [The SemEval-2016 Stance Dataset](http://www.saifmohammad.com/WebDocs/stance-data-all-annotations.zip).
An interactive visualization and further information about the dataset
can be found [here](http://www.saifmohammad.com/WebPages/StanceDataset.htm)

This repo explores a transfer-learning approach to Stance detection:

Uses a concatenation of the `tweets` encoded with
[LASER](https://github.com/facebookresearch/LASER) and one-hot encoded
 `targets` as inputs to different classifiers (`mlp`, `svm` or `random forest`).

No `fine-tunning`, `data-augmentation`, `hyper-parameter search` or
any other optimization technique is used.

This open the door to a variaty of tricks that could be applied to improve
results. Nevertheless we can see competitives results.


## Structure

```bash
# tree -L 3 -I "*.pyc|*cache*|*init*"
.
├── data
├── external
│   ├── encoders        # LASER encoding code
│   │   └── laser.py
│   ├── models          # pre-trained LASER encoder model
│   │   └── LASER
│   └── pyBPE           # BPE - Semantic Hash (or Subword Units) lib
├── README.md
├── requirements.txt
├── scripts
│   ├── download_models.sh
│   ├── eval.pl
│   ├── explore_dataset.ipynb
│   └── stance-detection-data-processing.html
├── setup.cfg
├── stance              # Stance package and relevant code
│   ├── data_utils
│   │   ├── loaders.py
│   │   ├── stance_batcher.py
│   │   └── text_processing.py
│   ├── encoders.py
│   └── laser_classifier.py     # main entry-point
├── tests
└── workdir             # directory with checkpoints and other artifacts
    ├── laser_mlp.pkl
    ├── predictions.csv
    ├── stance_laser-text+onehot-target.npy
    ├── test_laser-tweet+onehot-target.npy
    ├── test_laser-tweets.npy
    ├── test_laser-tweet+target.npy
    ├── training_laser-tweet+onehot-target.npy
    ├── training_laser-tweets.npy
    └── training_laser-tweet+target.npy
```


## How To:

### Prepare:

1. `./scripts/download_models.sh`
2. `pip install -r requirements`


### Training:

1. This trains a classifier of choice
while [LASER](https://github.com/facebookresearch/LASER)-encoding
the tweets and OneHot encoding the Stance:
```bash
    python -m stance.laser_classifier  train \
        --train-file data/SemEval2016-Task6-subtaskA-traindata-gold.csv \
        --test-file data/SemEval2016-Task6-subtaskA-testdata-gold.txt \
        --predictions-file ./results/semeval_predictions_mlp.csv \
        --clf-save-name laser \
        --clf-type mlp \
        --debug
```

After a while you should see an output similar to:
```bash
    Timeit - 'encode_or_load_data' took 0.3740 seconds
    DEBUG - laser_classifier.py - 142: x-train: (2914, 1029) | y-train: (2914,)
    DEBUG - laser_classifier.py - 144: x-test: (1249, 1029) | y-test: (1249,)
    DEBUG - laser_classifier.py - 145: Target names: ['AGAINST' 'FAVOR' 'NONE']
    DEBUG - laser_classifier.py - 168: Predictions: (1249,)
    INFO - laser_classifier.py - 172: Test score: 0.6413130504403523
                    precision    recall  f1-score   support

        AGAINST       0.72      0.78      0.75       715
        FAVOR         0.58      0.48      0.52       304
        NONE          0.44      0.43      0.43       230

    micro avg          0.64      0.64      0.64      1249
    macro avg          0.58      0.56      0.57      1249
    weighted avg       0.63      0.64      0.64      1249

    Timeit - 'make_classifier_and_predict' took 23.9830 seconds
```

Once the encoding and training finishes produces a `csv` file
than can be passed onto a `perl` script provided by the
[SemEval site](http://alt.qcri.org/semeval2016/task6/index.php?id=data-and-tools)
```bash
    perl ./scripts/eval.pl \
        ./data/SemEval2016-Task6-subtaskA-testdata-gold.txt \
        ./results/semeval_predictions_mlp.csv
```

Which should output a similar summary as at training time but only for
stances `FAVOR` and `AGAINST`:

```bash
============
Results
============
FAVOR     precision: 0.5800 recall: 0.4770 f-score: 0.5235
AGAINST   precision: 0.7206 recall: 0.7790 f-score: 0.7487
------------
Macro F: 0.6361
```

For a RandomForest classifier:
```bash
    perl ./scripts/eval.pl \
        ./data/SemEval2016-Task6-subtaskA-testdata-gold.txt \
        ./results/semeval_predictions_randomforest.csv

============
Results
============
FAVOR     precision: 0.8345 recall: 0.3816 f-score: 0.5237
AGAINST   precision: 0.6718 recall: 0.9678 f-score: 0.7931
------------
Macro F: 0.6584
```

In addition, all encoded inputs and models will be saved in the specified
`workdir`.
By default an `mlp` classifier will be saved at: `./workdir/laser_mlp.pkl`.
Similarly a `random forest` classifier: `./workdir/laser_randomforest.pkl`.

### Eval:
```bash
    python -m stance.laser_classifier  eval \
        --train-file data/SemEval2016-Task6-subtaskA-traindata-gold.csv \
        --test-file data/SemEval2016-Task6-subtaskA-testdata-gold.txt \
        --clf-save-name laser \
        --clf-type mlp \
        --debug
```

### Evaluation on a different domain:
```bash
    python -m stance.laser_classifier transfer \
        --test-file data/stance.csv \
        --predictions-file ./workdir/stance_predictions.csv
        --debug
```


## Notes:

* Dataset imbalance
* Control random seed
* Colab notebook
* Unit tests
* Typing
* Save models
* TF logs
* LaTex Doc
