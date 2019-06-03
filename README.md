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
```bash
    python run.py \
        --train-file data/SemEval2016-Task6-subtaskA-traindata-gold.csv \
        --test-file data/SemEval2016-Task6-subtaskA-testdata-gold.txt
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
