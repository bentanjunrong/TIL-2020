# TIL 2020 Team Mind Games notebooks

All our notebooks are stored here. Read the stuff below to acquaint urself with the workflow. Remember to read the [Workflow](#workflow) segment!

- [TIL 2020 Team Mind Games notebooks](#til-2020-team-mind-games-notebooks)
- [Using `pipenv`](#using-pipenv)
  - [Getting Started](#getting-started)
  - [Installing/Managing Libraries/Dependencies](#installingmanaging-librariesdependencies)
  - [Running Programs](#running-programs)
  - [Running Jupyter](#running-jupyter)
- [Workflow](#workflow)
  - [Dataset/Directory standardization](#datasetdirectory-standardization)
  - [Saving models](#saving-models)
    - [Installing Git LFS](#installing-git-lfs)


# Using `pipenv`
## Getting Started
1. Install `pipenv` with `pip3 install pipenv`
2. Clone this repository
3. Run `pipenv install` to pull dependencies
4. (Windows only) Special module for Windows users
   1. Enter shell with `pipenv shell`
   2. Run `pip install pywin32`
5. Setup [git LFS](#installing-git-lfs)
   
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

## Dataset/Directory standardization
 I am not gonna push the dataset to Github. For the sake of standardizing paths, create a folder called `data` and extract the contents of the [`Download All` zip file](https://www.kaggle.com/c/til2020/data?select=TIL_NLP_train_dataset.csv) here. I have also created a `saved_models` folder within the two parts (CV and NLP) to save the models we have trained locally. [Refer to the next section for more info](#saving-models)
 
 
  Here is how the overall directory looks on my pc. Folders that are not pushed to the Github repo are marked with *

 ```
.
├── CV
│   ├── CV-Notebooks-go-here!
│   ├── ravyu_RESNET50_defaulttemplate.ipynb
│   └── saved_models
├── data*
│   ├── NLP_submission_example.csv
│   ├── resnet50_weights_tf_dim_ordering_tf_kernels.h5
│   ├── resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
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

9 directories, 15 files
```

## Saving models
If you want to save any trained models you can do so. These models are usually large in size (mine was 90+ mb), ~~so do not push these to GitHub. I will figure out some other way to share these files. ~~ Ok, let's use [git lfs](https://git-lfs.github.com/)

### Installing Git LFS
1. Follow step 1 of getting started [from the Git LFS website](https://git-lfs.github.com/)
2. That's it! I have already setup the repo to push the folders CV/saved_models and NLP/saved_models(WIP) through LFS. So just push and pull as per normal.

