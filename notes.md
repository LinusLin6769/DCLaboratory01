# Structure of experiment

A mother config.json file is provided which contains all the predefined functionalities and options for the experiment. To initiate a new run, make a copy of the mother config.json file, edit it and make it a child config.json file for this new run you are having. Everything regarding this new run will be stored in a newly created directory, including the child config.json file (but, obviously, not the mother config.json file).

# Configuration

Things to be specified in the config.json file:

Dataset:

- path the the csv
- csv: 2d array with multiple time series, say, n time series
- how many time seies you want to use in this run
    - randomly chosen from n time series
    - provide and array of indices
    - the first m time series
    - the last m time series

- validation size
- test size
- number of lags
- forecast horizon

Transformation options:

- threshold:
    - single user define
    - explore
        - what is the range
        - how many trials
            - grid search
            - randon search

Model options:

- a list of models to try
    - for each model, which hyperparameters should be tuned

Output options:

- create a directory and put everything in it
    - a json file for each candidate
    - a copy of the config.json file
    - a log of the prompted messages (e.g., warnings)
    - a metadata.json file of additional measurements of the experiment:
        - the original time series
        - thresholds found via tuning
        - markers for making plots
        - movement values for directional changes and overshoots
        - hopefully, duration values for directional changes and overshoots (number of data points?)
        - the status array
        - hyperparameters for the models
        - y_hat (the predicted values)

# Anaylsis of experiment

Analyses are done separately. After the run is completed, you should have a lot of data generated and stored. A separeted analysis procedure is then taken using the generated data. In this phase, this could be done with a jupyter notebook or additional python scripts. Since the generated data is stored in JSON format, any preferred analysis tool should be able to be performed.

My analysis:

- compute SMAPE using the testing results
- compute the ranking using the SMAPE
