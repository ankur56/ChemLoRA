#!/usr/bin/env python3

from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from loguru import logger

import gptchem
from gptchem.tuner import Tuner
from gptchem.formatter import RegressionFormatter
from gptchem.querier import Querier
from gptchem.extractor import RegressionExtractor

with open("qm9_key_smiles_0_val_u0_atom_b3lyp.pickle", "rb") as handle:
        qm9_data = pickle.load(handle)

qm9_data_pd = pd.DataFrame(list(qm9_data.items()), columns=["SMILES", "B3LYP atomization energy in kcal/mol"])

# encode the data into prompts and completions
formatter = RegressionFormatter(representation_column='SMILES',
    label_column='B3LYP atomization energy in kcal/mol',
    property_name='atomization energy in kcal/mol',
    num_digits=4
    )

formatted_data = formatter.format_many(qm9_data_pd)

# split the data into training and test set
train, test = train_test_split(formatted_data, test_size=0.1, random_state=42)

# enable gptchem logging
logger.enable("gptchem")

# define the logging level
LEVEL = "DEBUG"
logger.add("my_log_file.log", level=LEVEL, enqueue=True)

# initialize the tuner
tuner = Tuner()
tune_summary = tuner(train)

print(tune_summary)

# initialize the querier
querier = Querier('ada') # use the model called 'ada'

# get the completions (assuming the test frame we created above)
completions = querier(test)

# extract the predictions
extractor = RegressionExtractor()
predictions = extractor(completions)
print(predictions)
metrics = gptchem.evaluator.get_regression_metrics(predictions, test['label'])
print(metrics)

