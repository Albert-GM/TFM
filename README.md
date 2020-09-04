TFM
==============================

Master's Thesis


## Want to install this project on your own computer?

Start by installing Anaconda (or Miniconda) and git.

Next, clone this project by opening a terminal and typing the following commands (do not type the first $ signs on each line, they just indicate that these are terminal commands):

```
$ git clone https://github.com/Albert-GM/TFM.git
$ cd TFM
```

Next, run the following commands:

```
$ conda env create -f environment.yml
$ conda activate tfm-agm
```


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README with project description and reproducibility steps.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │    
    ├── models             <- Trained and serialized models
    │
    ├── environment.yml   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `conda env create --file environment.yml`
    │
    └── src                <- Source code for use in this project.
        │
        ├── data           <- Scripts to generate data
        │  
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │  
        │
        └── models         <- Scripts to train models and then use trained models to make
                              predictions
