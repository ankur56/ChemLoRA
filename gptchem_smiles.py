#!/usr/bin/env python3

import json
import pickle

from gptchem.gpt_regressor import GPTRegressor
from gptchem.tuner import Tuner


def pickle_to_data(filename):
    with open(filename, "rb") as handle:
        qm9_data = pickle.load(handle)

    # Convert the keys and values of the dictionary into separate lists
    smiles_list = list(qm9_data.keys())
    property_list = list(qm9_data.values())

    # Extract the B3LYP atomization energy as a separate list
    b3lyp_at_list = [prop[0] for prop in property_list]

    # Extract the G4MP2 atomization energy as a separate list
    g4mp2_at_list = [prop[1] for prop in property_list]

    # Extract the (G4MP2-B3LYP) atomization energy difference as a separate list
    en_diff_list = [prop[2] for prop in property_list]

    # Extract the bandgap as a separate list
    #bandgap = [prop[3] for prop in property_list]

    return smiles_list, b3lyp_at_list, g4mp2_at_list, en_diff_list


if __name__ == '__main__':
    train_smiles_list, train_b3lyp_at_list, train_g4mp2_at_list, train_en_diff_list = pickle_to_data(
        "pickles/qm9_key_smiles_1_full_train_data.pickle")
    holdout_smiles_list, holdout_b3lyp_at_list, holdout_g4mp2_at_list, holdout_en_diff_list = pickle_to_data(
        "pickles/qm9_key_smiles_1_holdout_data.pickle")
    #sugar_smiles_list, sugar_b3lyp_at_list, sugar_g4mp2_at_list, sugar_en_diff_list = pickle_to_data("pickles/sugar_data_key_smiles_full.pickle")

    regressor = GPTRegressor(
        #property_name="G4MP2 and B3LYP atomization energy difference in kcal/mol", # this is the property name we will use in the prompt template
        property_name=
        "G4MP2 atomization energy in kcal/mol",  # this is the property name we will use in the prompt template
        tuner=Tuner(n_epochs=8,
                    learning_rate_multiplier=0.02,
                    wandb_sync=False),
    )

    # Fit the regressor with the train set
    regressor.fit(train_smiles_list, train_g4mp2_at_list)
    #regressor.fit(train_smiles_list, train_en_diff_list)

    # Make predictions using the holdout set
    y_pred_holdout = regressor.predict(holdout_smiles_list)
    y_pred_holdout_list = y_pred_holdout.tolist()
    y_true_holdout = holdout_g4mp2_at_list
    #y_true_holdout = holdout_en_diff_list

    #y_pred_sugar = regressor.predict(sugar_smiles_list)
    #y_pred_sugar_list = y_pred_sugar.tolist()
    #y_true_sugar = sugar_g4mp2_at_list
    #y_true_sugar = sugar_en_diff_list

    results = {
        'y_pred_holdout': y_pred_holdout_list,
        'y_true_ho': y_true_holdout
    }
    #results = {'y_pred_holdout': y_pred_holdout_list, 'y_true_ho': y_true_holdout, 'y_pred_sugar': y_pred_sugar_list, 'y_true_sugar': y_true_sugar}

    with open('gptchem_results_smiles_g4mp2_y_all.json', 'w') as f:
        #with open('gptchem_results_smiles_en_diff_y_all.json', 'w') as f:
        json.dump(results, f)
