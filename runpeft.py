#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from gptchem.gpt_classifier import GPTClassifier
from gptchem.tuner import Tuner
from gptchem.formatter import RegressionFormatter
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt

from scipy.stats import pearsonr
import pickle
import json
TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
}
import sys
from typing import List
import torch
import transformers

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel,
)
from datasets import Dataset
import torch
#from utils.callbacks import Iteratorize, Stream

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, EarlyStoppingCallback,GenerationConfig
import os
print(torch.cuda.device_count())
#get the data
def get_data(prop_to_get):
    
    prop = {"b3lyp": "B3LYP atomization energy in kcal/mol",
            "g4mp2": "G4MP2 atomization energy in kcal/mol",
            "en_diff": "atomization energy difference in kcal/mol",
            "bandgap": "bandgap in Hartrees"}
    
    def pickle_to_df(filename):
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
        bandgap = [prop[3] for prop in property_list]

        df = pd.DataFrame(list(zip(smiles_list, b3lyp_at_list, g4mp2_at_list,en_diff_list,bandgap)),
                   columns =["SMILES", prop["b3lyp"], prop["g4mp2"], prop["en_diff"], prop["bandgap"]])
        return df
    
    #unpickle the data
    train_df = pickle_to_df("pickles/qm9_key_smiles_1_train_data_without_validation.pickle")
    val_df = pickle_to_df("pickles/qm9_key_smiles_1_validation_data.pickle")
    test_df = pickle_to_df("pickles/qm9_key_smiles_1_holdout_data.pickle")
    
    #format the data as text for the LLM
    formatter = RegressionFormatter(representation_column='SMILES',
        label_column=prop[prop_to_get],
        property_name=prop[prop_to_get],
        num_digits=4
        )
    
    df_train = formatter.format_many(train_df).drop(columns=["label","representation"], axis=1)
    df_val = formatter.format_many(val_df).drop(columns=["label","representation"], axis=1)
    df_test = formatter.format_many(test_df).drop(columns=["label","representation"], axis=1)
    
    return df_train, df_val, df_test
def train(
    #dataframes
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    # model/data params
    base_model: str = "gpt2",  # the only required argument
    prop: str = "b3lyp",
    output_dir: str = "outputs",
    # training hyperparams
    batch_size: int = 256,
    num_epochs: int = 20,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [""],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "chemgpt",  # The prompt template to use, will default to alpaca.
):
    
    lora_target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[base_model]
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"prop: {prop}\n"
            f"output_dir: {output_dir}_{prop}_{base_model}\n"
            f"batch_size: {batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='gpt2'"
    #
    #gradient_accumulation_steps = 1

    #tells us how to split the model between GPUs 
    #device_map = "sequential"
    #world_size = int(os.environ.get("WORLD_SIZE", 1))
    #ddp = world_size != 1
    #ddp = False
    #if ddp:
    #    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    #    gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    #set up the model and tokenizer    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    #get the model with the desired name automatically
    model = AutoModelForCausalLM.from_pretrained(
        base_model, 
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map='sequential',
    )    
    #tokenizer settings
    def tokenize(prompt):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=True,
            return_tensors=None,
        )
        return result
    #tokenize the full prompt
    def tokenize_prompt(data_point):
        full_prompt = data_point["prompt"]+data_point["completion"]
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    train_data = Dataset.from_pandas(df_train).shuffle().map(tokenize_prompt)
    val_data = Dataset.from_pandas(df_val).shuffle().map(tokenize_prompt)
    
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    print(train_data[0])
    print(val_data[0])
    """
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    """
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            warmup_steps=10,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            output_dir=output_dir+"_"+prop+"_"+base_model,
            save_total_limit=3,
            metric_for_best_model = 'eval_loss',
            load_best_model_at_end=True,
            #ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    print(old_state_dict)
    print(model.state_dict)
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir+"_"+prop+"_"+base_model)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

def generate(
    df_test: pd.DataFrame,
    load_8bit: bool = False,
    base_model: str = "gpt2",
    lora_weights: str = "outputs",
    prop: str = "b3lyp",
    prompt_template: str = "",
    cutoff_len: int = 256,
):

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu" 

    #set up the model and tokenizer    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # might not be optimal, just trying to run the code
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize(prompt):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=True,
            return_tensors="pt",
        )
        return result

    model= AutoModelForCausalLM.from_pretrained(
        base_model, 
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map='auto',
    )    
    model = PeftModel.from_pretrained(
        model,
        lora_weights+"_"+prop+"_"+base_model,
        torch_dtype=torch.float16,
    )

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        prompt,
        temperature=0,
        top_p=0.75,
        top_k=40,
        num_beams=2,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):

        inputs = tokenize(prompt)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        # Generate outs without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        #print(output)
        return output

    #convert to a number if we can, else none
    def toFloat(x):
        try:
            return float(x)
        except:
            return None 
    
    df_test["model_out"] = df_test["prompt"].map(lambda x: evaluate(x))
    df_test["energy_out"] = df_test["model_out"].map(lambda x: toFloat(x.replace('###','@@@').split('@@@')[1]))
    df_test["energy_true"] = df_test["completion"].map(lambda x: toFloat(x.split('@@@')[0]))
    return(df_test)

def run_all(model):
    df_train,df_val,df_test = get_data("b3lyp")
    train(df_train, df_val, base_model=model, prop="b3lyp")
    outs=generate(df_test.head(1000), base_model=model, prop="b3lyp")
    outs.to_json(f"outputs_{model}_b3lyp.json")
    outs=outs.dropna()
    plt.scatter(outs['energy_true'], outs['energy_out'])
    pearsonr(outs['energy_true'], outs['energy_out'])
    plt.axis('square')
    plt.savefig(f"outputs_{model}_b3lyp.png")
    plt.clf()

    df_train,df_val,df_test = get_data("g4mp2")
    train(df_train, df_val, base_model=model, prop="g4mp2")
    outs=generate(df_test.head(1000), base_model=model, prop="g4mp2")
    outs.to_json(f"outputs_{model}_g4mp2.json")
    outs=outs.dropna()
    plt.scatter(outs['energy_true'], outs['energy_out'])
    pearsonr(outs['energy_true'], outs['energy_out'])
    plt.axis('square')
    plt.savefig(f"outputs_{model}_g4mp2.png")
    plt.clf()

    df_train,df_val,df_test = get_data("en_diff")
    train(df_train, df_val, base_model=model, prop="en_diff")
    outs=generate(df_test.head(1000), base_model=model, prop="en_diff")
    outs=outs.dropna()
    outs.to_json(f"outputs_{model}_en_diff.json")
    plt.scatter(outs['energy_true'], outs['energy_out'])
    pearsonr(outs['energy_true'], outs['energy_out'])
    plt.axis('square')
    plt.savefig(f"outputs_{model}_endiff.png")
    plt.clf()

    df_train,df_val,df_test = get_data("bandgap")
    train(df_train, df_val, base_model=model, prop="bandgap")
    outs=generate(df_test.head(1000), base_model=model, prop="bandgap")
    outs.to_json(f"outputs_{model}_bandgap.json")
    outs=outs.dropna()
    plt.scatter(outs['energy_true'], outs['energy_out'])
    pearsonr(outs['energy_true'], outs['energy_out'])
    plt.axis('square')
    plt.savefig(f"outputs_{model}_bandgap.png")
    plt.clf()

if __name__ == '__main__':
     run_all("gpt2")










