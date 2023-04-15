#!/usr/bin/env python3

import pickle
import json

with open('g4mp2_data.json') as f:
    data = json.loads("[" + f.read().replace("}\n{", "},\n{") + "]")
    red_data = {}
    for d in data:
        mol_smiles = d['smiles_0'] 
        g4mp2_at = float(d['g4mp2_atom'])*627.509
        b3lyp_at = float(d['u0_atom'])*627.509
        red_data[mol_smiles] = b3lyp_at
    with open("qm9_key_smiles_0_val_u0_atom_b3lyp.pickle", "wb") as handle:
        pickle.dump(red_data, handle, protocol=pickle.HIGHEST_PROTOCOL)





