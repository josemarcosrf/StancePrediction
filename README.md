# Stance Detection


## Dataset

We use [The SemEval-2016 Stance Dataset](http://www.saifmohammad.com/WebDocs/stance-data-all-annotations.zip).
An interactive visualization and further information about the dataset
can be found [here](http://www.saifmohammad.com/WebPages/StanceDataset.htm)


## Structure

```bash
# tree -L 3 -I "*.pyc|*cache*|*init*"
.
├── Factmata ML Test.pdf
└── stance_detection
    ├── data
    │   ├── SemEval2016-Task6-subtaskA-testdata-gold.txt
    │   ├── SemEval2016-Task6-subtaskA-traindata-gold.csv
    │   └── stance.csv
    ├── external
    │   ├── encoders
    │   ├── models
    │   ├── pyBPE
    │   └── workdir
    ├── README.md
    ├── run.py
    ├── scripts
    │   ├── explore_dataset.ipynb
    │   └── stance-detection-data-processing.html
    ├── stance
    │   ├── classifiers
    │   ├── data_utils
    │   └── training
    └── tests


```


## How To:

### Prepare:

1. clone [pyBPE](https://github.com/jmrf/pyBPE) into `./external`

2. [...]


### Training:

1. Traing a classifier [LASER](https://github.com/facebookresearch/LASER)-encoding
the tweets and OneHot encoding the Stance:
```bash
    python -m stance.laser_classifier  \
        --train-file data/SemEval2016-Task6-subtaskA-traindata-gold.csv \
        --test-file data/SemEval2016-Task6-subtaskA-testdata-gold.txt \
        --debug
```

This should output something similar to:
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

### Eval:
```bash
    python run.py eval \
        --eval-file data/stance.csv
        --ckpt-file ...
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
