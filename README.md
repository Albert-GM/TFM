PREDICTING THE SCOPE OF A PANDEMIC
==============================
Master's thesis
------------

Below you can find a summary of the thesis, the steps to reproduce the results of the work and the project organization.


Summary
------------

Research on disease spread has a long tradition, and most are based on stochastic or deterministic mathematical models. These models are based on probabilities or initial conditions to determine the likely outcome of a pandemic. Machine Learning models have rarely been used to predict that outcome. It may be due to the lack of pandemic data since these models are based on learning from large amounts of them. This work suggests the use of a mathematical model to simulate a considerable amount of data about the spread of different diseases and use that information to feed a machine learning model. Of this way, the ML model is provided with enough power to learn and make accurate predictions.

In particular, with this aim, I create a deterministic mathematical model based on the well-known SIR model. The model is adapted to a global pandemic situation, allowing interactions between different countries and the quarantine of each one if needed. The necessary data for the mathematical model are collected from public sources and include information about countries, as total population, number of international arrivals or departures. Moreover, I use data from airports around the world to know which routes exist between countries. All of these data allow creating an origin-destination matrix, simulating the movement of individuals between countries and being able to identify which countries are the most travelled and therefore, the primary sources of spread of the disease. Also, linking the information to a bidirectional graph, I compute features based on graph theory for each country, such as degree or betweenness.

Next, I explore a wide range of parameters, mixing different types of diseases, different focal points of the pandemic and quarantine intensities. In this way, the mathematical model can simulate distinct pandemic situations. Once the data are generated, they are provided to the ML model. It should be noted that the features used in the ML model do not include any characteristic parameter of the SIR model, that cannot be calculated or are challenging to compute in a real pandemic situation. Furthermore, I use data generated in the first two weeks after the first deceased, to predict the outcome of the pandemic. So the input parameters of the ML model are not the same as those of the mathematical model, which makes the learning process more challenging.

Finally, after the corresponding exploratory analysis of the data simulated, I train different ML models, getting a satisfactory result with tree-based models and a neural network. The models that best suit the dataset are a extreme gradient boosting that let explains the 93 % of the variability observed in the response variable and a deep neural network that is capable of explaining the 94 % of the variability. Therefore this work concludes with an acceptable result, being able to identify the patterns that produce a pandemic outbreak.


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

To train the ML models, first, it is necessary to execute add_features.py that produces the dataset from the simulation results.

If you want to train the neural network in the project, it is advisable to create a separate environment with Tensorflow and Keras libraries.


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README with project description and reproducibility steps.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │   
    │── figures            <- Figures used in the work.
    │
    ├── models             <- Trained and serialized models.
    │
    ├── environment.yml    <- The requirements file for reproducing the analysis environment
    │                     `
    ├── data_flow.png      <- Scheme of the data transformation process.
    │
    ├── thesis.pdf        <- Thesis.    
    │
    └── src                <- Source code for use in this project.
        │
        ├── data           <- Scripts to generate data.
        │  
        │
        ├── features       <- Scripts to turn raw data into features for modeling.
        │  
        │
        └── models         <- Scripts to train models and then use trained models to make
                              predictions.
