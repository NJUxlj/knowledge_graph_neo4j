# -*- coding: utf-8 -*-

Config = {
    "model_path": "model_output",
    "train_data_path": "triplet_data/train_triplet_data.json",
    "valid_data_path": "triplet_data/valid_triplet_data.json",
    "pretrain_model_path":r"D:\models\bert-base-chinese",
    "schema_path":"Rbert/schema.json",
    "max_length": 100,
    "epoch": 10,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 1e-5,
}