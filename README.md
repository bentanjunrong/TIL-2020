# Using pipenv
## Getting Started
1. Install `pipenv` with `pip3 install pipenv`
2. Clone this repository
3. Run `pipenv install` to pull dependencies (so far it's just pandas)
   
## Installing/Managing Libraries/Dependencies
- Install dependencies for this repository with `pipenv install XXX` and inform everybody of any new dependencies you have pushed into the repo
- When pulling new commits from the repo, its a good practice to run `pipenv update` to ensure you pull any new dependencies
- Remove dependencies with pipenv uninstall (make sure you push the changes)

## Running Programs
1. Run the code using `pipenv run XXX.py` or enter shell with `pipenv shell` and then execute the file
   - This runs the code in the virtual environment in which all these dependencies were installed to (you would have seen its location when u first ran `pipenv install` in getting started)


## Running Jupyter
1. Following the [previous section](#running-programs), you run JupyterLab by:
   1. Enter shell mode with `pipenv shell`
   2. Run `jupyter lab`
2. I take reference to [this guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html)

Instructions based off of [this guide](https://pipenv.pypa.io/en/latest/basics/#general-recommendations-version-control)

# Workflow

When you pull dependancies you should have everything you need to run the notebooks.
I have created folders for each part of the competition (CV and NLP). Save your notebooks there

## Dataset
 I am not gonna push the dataset to Github. For the sake of standardizing paths, create a folder called `data` and extract the contents of the [`Download All` zip file](https://www.kaggle.com/c/til2020/data?select=TIL_NLP_train_dataset.csv) here. Here is how the overall directory looks on my pc

 ```
 .
├── CV
│   └── CV-Notebooks-go-here!
├── data
│   ├── NLP_submission_example.csv
│   ├── TIL_NLP_test_dataset.csv
│   ├── TIL_NLP_train_dataset.csv
│   ├── train
│   ├── train.json
│   ├── train.p
│   ├── val
│   ├── val.json
│   ├── val.p
│   └── word_embeddings.pkl
├── docs
│   └── TIL 2020 Qualifier Info Pack v1 0.pdf
├── NLP
│   └── NLP-Notebooks-go-here!
├── Pipfile
├── Pipfile.lock
└── README.md

8 directories, 12 files
```