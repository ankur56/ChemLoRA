# ChemLoRA

## Leveraging LLMs for Accurate Molecular Energy Predictions

### Requirements
- [gptchem](https://github.com/kjappelbaum/gptchem)
- [transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [datasets](https://github.com/huggingface/datasets)
- [scikit-learn](https://scikit-learn.org/stable/)

### Data
The QM9-G4MP2 dataset is publicly available through [Materials Data Facility](https://petreldata.net/mdf/detail/wardlogan_machine_learning_calculations_v1.1/). 

### Model Fine-Tuning

GPT-3 is fine-tuned on the QM9-G4MP2 dataset using the GPTChem framework by running the given python script as follows,\
`python gptchem_smiles.py`

`runpeft.py` can be used to fine-tune any foundational LLM available in Hugging Face. For example, the following command could be used to fine-tune the `gpt2` model\
`python runpeft.py "gpt2"`

