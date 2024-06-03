# -*- coding: utf-8 -*-
import time
import pickle
import numpy as np
import nltk
import evaluate
import collections
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

import torch
# Load model directly
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DefaultDataCollator,
                            SwitchTransformersForConditionalGeneration,
                            AutoConfig, get_scheduler, DataCollatorForSeq2Seq,
                            )
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
# import evaluate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def save_model(model,name):
    save_path = dir_path + '/pth/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    # model_checkpoint = os.path.join(dir_path + '/pth/', "%s_checkpoint.bin" % name)
    torch.save(model_to_save.state_dict(), save_path+f'moe_{name}_checkpoint.bin')
    print(f"Saved model checkpoint to {save_path}!")

def Create_MoE_Model(**kwargs):
    # bert
    if kwargs['model_name'] == 'bert':
        from transformers import BertForQuestionAnswering
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-medium")
        config = AutoConfig.from_pretrained("prajjwal1/bert-medium")
        config_load = AutoConfig.from_pretrained("prajjwal1/bert-medium")
        config.moe = kwargs['moe']
        config.moe_num_experts = kwargs['moe_num_experts']
        config.moe_top_k = kwargs['moe_top_k']
        config.moe_group = kwargs['moe_group']
        config.moe_world_size = kwargs['moe_world_size']

        modelForLoad = BertForQuestionAnswering.from_pretrained("prajjwal1/bert-medium", config=config_load)
        if config.moe_num_experts == 0:
            return modelForLoad,tokenizer

        mymoe = BertForQuestionAnswering.from_pretrained("prajjwal1/bert-medium", config=config)
        # print(modelForLoad.state_dict().keys(),mymoe.state_dict().keys())
        mymoeParam = mymoe.state_dict()
        bertParam = modelForLoad.state_dict()
        # original weight = ['bert.encoder.layer.10.intermediate.dense.weight', 'bert.encoder.layer.10.intermediate.dense.bias', 
        # 'bert.encoder.layer.10.output.dense.weight', 'bert.encoder.layer.10.output.dense.bias']
        # desity weight = ['bert.encoder.layer.0.moe_linear.experts.0.htoh4.weight', 'bert.encoder.layer.0.moe_linear.experts.0.htoh4.bias', 
        # 'bert.encoder.layer.0.moe_linear.experts.0.h4toh.weight', 'bert.encoder.layer.0.moe_linear.experts.0.h4toh.bias',]
        # original_layer_normal = ['bert.encoder.layer.11.output.LayerNorm.weight', 'bert.encoder.layer.11.output.LayerNorm.bias']
        # desitny weight = ['bert.encoder.layer.0.moe_linear.layer_norm.weight', 'bert.encoder.layer.0.moe_linear.layer_norm.bias']
        # bertLayerLength=24
        # copy linear weight, bias and layernormal
        for layer in range(config_load.num_hidden_layers):
            for expert_id in range(config.moe_num_experts):
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.weight'] = bertParam['bert.encoder.layer.'+str(layer)+'.intermediate.dense.weight'].unsqueeze(0).detach().clone()
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.bias'] = bertParam['bert.encoder.layer.'+str(layer)+'.intermediate.dense.bias'].unsqueeze(0).detach().clone()
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.weight'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.dense.weight'].unsqueeze(0).detach().clone()
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.bias'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.dense.bias'].unsqueeze(0).detach().clone()
            mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.layer_norm.weight'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.LayerNorm.weight'].detach().clone()
            mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.layer_norm.bias'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.LayerNorm.bias'].detach().clone() 
        mymoe.load_state_dict(mymoeParam)
        return mymoe, tokenizer
    else:
        raise Exception('Error: no such a model named {}'.format(kwargs['model_name']))

