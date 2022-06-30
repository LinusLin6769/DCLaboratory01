# How the programme works

1. Set how the experiment is intended to be done using the `config.json` file.
2. Run the experiment, which will generate data regarding the experiment.
3. Analyse the generated data and come up with insights.

## config

The things that are controlled by the `config.json` file include

- whether this run is for real, or it's just a test of something
- which dataset to use, which series to use
- search space of the parameters of the transformation
- how the modelling process is done (validation size, test size etc.)
- which models to use
- how many CPU cores to use during multiprocessing

## Running the experiment

The experiment runs by calling

```{command line tool}
python3 main.py
```

from the command-line tool. The input is the `config.json` file, where the user controls how the experiment should go, and the output is a directory being created under the `experiment_info/` directory named after the starting time of the experiment.

The execution goes as the following:

- `main.py` gets the configuration of the experiment from `config.json`
- `main.py` performs some useful checks on whether the inputs it read from `config.json` are valid and reasonable
- `main.py` gets the search space of the hyperparameters of the models from `model_policies`
- `main.py` then starts training the models using the dataset specified in `config.json`, which is stored in the directory `datasets/`.

  - Note that the programme currently is only able to read .JSON files. This is one of the reasons why the `playground.ipynb` exists.
  - A model is trained on all the series used, a .JSON file containing the experiment information of this model is created and placed in the directory of this experiment, and the programme moves on to another model until all models are trained.

- After the models are trained, a dictionary of overall information regarding this run is appended to the `runs.json` file, which is a log of all the runs being run.

## Analysing the results

The analysis tool is the `run_analysis.ipynb` notebook. You can export the notebook to a .html file and put it in the experiment directory.
