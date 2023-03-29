#!/usr/bin/env python3

import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from gptchem.gpt_regressor import GPTRegressor
from gptchem.tuner import Tuner

with open("qm9_key_smiles_0_val_u0_atom_b3lyp.pickle", "rb") as handle:
        qm9_data = pickle.load(handle)

# Convert the keys and values of the dictionary into separate lists
smiles_list = list(qm9_data.keys())
energies_list = list(qm9_data.values())

# Split the dataset into train (90%) and test (10%) sets
train_smiles, test_smiles, train_energies, test_energies = train_test_split(
    smiles_list, energies_list, test_size=0.1, random_state=42
)

regressor = GPTRegressor(
    property_name="atomization energy in kcal/mol", # this is the property name we will use in the prompt template
    tuner=Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False),
)

# Fit the regressor with the train set
regressor.fit(train_smiles, train_energies)

# Make predictions using the test set
y_pred = regressor.predict(test_smiles)
y_true = test_energies

# Calculate the regression metrics between predictions and test_energies
regression_metrics = {
    "r2": r2_score(y_true, y_pred),
    "max_error": max_error(y_true, y_pred),
    "mean_absolute_error": mean_absolute_error(y_true, y_pred),
    "mean_squared_error": mean_squared_error(y_true, y_pred),
    "rmse": mean_squared_error(y_true, y_pred, squared=False),
    "mean_absolute_percentage_error": mean_absolute_percentage_error(y_true, y_pred),
    }

print(regression_metrics)
