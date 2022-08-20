from cmath import exp
import os
import torch
import json

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True


FP_PREPROCESSED_TRAIN_CSV = "./intermediary_state/train.csv"
FP_PREPROCESSED_TEST_CSV = "./intermediary_state/test.csv"
FP_PREPROCESSED_VAL_CSV = "./intermediary_state/val.csv"

FP_ORIGINAL_TRAIN_CSV = "./data/train.csv"
FP_ORIGINAL_TEST_CSV = "./data/test.csv"

FP_ORIGINAL_TRAIN_ESSAY_DIR = "./data/train/"
FP_ORIGINAL_TEST_ESSAY_DIR = "./data/test/"

FP_OUTPUT_SAMPLE_SUBMISSION_CSV = "./data/submission_fullset.csv"

FP_PREPROCESSED_BY_CATEGORY_CSV_DIR = "./intermediary_state/by_category"

FP_MERGED_FILE_OUTPUT = "./data/merged_file.txt"

FP_GENERATED_DIR = "./generated_content"

FP_EXPERIMENT_PARAMETERS_DIR = "./experiment_parameters"
FP_DEBERTA_MODEL_CONFIG = "./deberta_current_config.json"


TRAINED_MODELS = {
    "bert-base-uncased": "./trained_models/bert-uncased",
    "microsoft/deberta-v3-large": "./trained_models/microsoft-deberta-v3-large",
    "microsoft/deberta-v3-base": "./trained_models/microsoft-deberta-v3-base",
    "microsoft/deberta-v3-xsmall": "./trained_models/microsoft-deberta-v3-xsmall",
    "microsoft/deberta-v3-small": "./trained_models/microsoft-deberta-v3-small",
    "microsoft/deberta-v2-xlarge-mnli": "./trained_models/microsoft/deberta-v2-xlarge-mnli",
    "gpt2": "./trained_models/gpt2-text-generation",
}

BY_CATEGORY_TRAINED_MODEL_DIR = "by_category"
# MODEL_NAME_IN_USE = "bert-base-uncased"
# EXPERIMENT_SUFFIX = ""


# MODEL_NAME_IN_USE = "microsoft/deberta-v3-large"
# EXPERIMENT_SUFFIX = ""

# MODEL_NAME_IN_USE = "microsoft/deberta-v3-base"
# EXPERIMENT_SUFFIX = ""

# MODEL_NAME_IN_USE = "microsoft/deberta-v3-xsmall"
# EXPERIMENT_SUFFIX = ""

MODEL_NAME_IN_USE = "microsoft/deberta-v3-small"
EXPERIMENT_SUFFIX = ""


TRAINED_MODEL_KEY = MODEL_NAME_IN_USE + EXPERIMENT_SUFFIX

FP_TRAINED_MODEL_IN_USE = TRAINED_MODELS[TRAINED_MODEL_KEY]
CHECKPOINT_DIR = f"./checkpoint_{TRAINED_MODEL_KEY}/"

for dir_ in [FP_TRAINED_MODEL_IN_USE, CHECKPOINT_DIR, FP_EXPERIMENT_PARAMETERS_DIR]:
    try:
        os.makedirs(dir_)
    except FileExistsError:
        pass

REMOVE_STOP_WORDS = True
USE_CONTEXT = True

TEST_SIZE = 0.2
VAL_SIZE = 0.2
NUM_LABELS = 3
TOKENIZER_MAX_SIZE = 2048  # 512, 1024 or 2048
HIDDEN_ACT = "gelu"
HIDDEN_DROPOUT_PROB = 0.3
ATTENTION_PROBS_DROPOUT_PROB = 0.4
POOLER_DROPOUT = 0.4
LAYER_NORM_EPS = 1e-12

TRAINING_CONFIG_TO_SAVE = {
    "_name_or_path": "microsoft/deberta-v3-small",
    "architectures": ["DebertaV2ForSequenceClassification"],
    "attention_probs_dropout_prob": ATTENTION_PROBS_DROPOUT_PROB,
    "hidden_act": HIDDEN_ACT,
    "hidden_dropout_prob": HIDDEN_DROPOUT_PROB,
    "hidden_size": 768,
    "id2label": {"0": "ADEQUATE", "1": "EFFECTIVE", "2": "INEFFECTIVE"},
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "label2id": {"ADEQUATE": 0, "EFFECTIVE": 1, "INEFFECTIVE": 2},
    "layer_norm_eps": LAYER_NORM_EPS,
    "max_position_embeddings": TOKENIZER_MAX_SIZE,
    "max_relative_positions": -1,
    "model_type": "deberta-v2",
    "norm_rel_ebd": "layer_norm",
    "num_attention_heads": 12,
    "num_hidden_layers": 6,
    "pad_token_id": 0,
    "pooler_dropout": POOLER_DROPOUT,
    "pooler_hidden_act": "gelu",
    "pooler_hidden_size": 768,
    "pos_att_type": ["p2c", "c2p"],
    "position_biased_input": False,
    "position_buckets": 256,
    "relative_attention": True,
    "share_att_key": True,
    "torch_dtype": "float32",
    "transformers_version": "4.21.1",
    "type_vocab_size": 0,
    "vocab_size": 128100,
}
with open(FP_DEBERTA_MODEL_CONFIG, "w") as fp:
    json.dump(TRAINING_CONFIG_TO_SAVE, fp)

METRIC = "accuracy"

from transformers import TrainingArguments

training_args_raw = {
    "output_dir": CHECKPOINT_DIR,
    "report_to": None,
    "evaluation_strategy": "steps",
    "num_train_epochs": 8,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0.0001,
    "data_seed": 0,
    "seed": 0,
    "logging_steps": 2000,
    "eval_steps": 2000,
    "save_steps": 2000,
    "adam_beta1": 0.9,
    "adam_beta2": 0.99,
    "adam_epsilon": 1e-8,
    "load_best_model_at_end": True,
    "gradient_accumulation_steps": 1,
    "bf16": True,
    "bf16_full_eval": True,
}
training_args = TrainingArguments(**training_args_raw)

experimental_args = {
    "Rebuttal": {
        "output_dir": CHECKPOINT_DIR,
        "report_to": None,
        "evaluation_strategy": "steps",
        "num_train_epochs": 8,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "learning_rate": 2e-5,
        "weight_decay": 0.0001,
        "data_seed": 0,
        "seed": 0,
        "logging_steps": 40,
        "eval_steps": 40,
        "save_steps": 40,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "adam_epsilon": 1e-8,
        "load_best_model_at_end": True,
        "gradient_accumulation_steps": 1,
        "bf16": True,
        "bf16_full_eval": True,
    },
    "Claim": {
        "output_dir": CHECKPOINT_DIR,
        "report_to": None,
        "evaluation_strategy": "steps",
        "num_train_epochs": 4,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "learning_rate": 2e-5,
        "weight_decay": 0.0001,
        "data_seed": 0,
        "seed": 0,
        "logging_steps": 200,
        "eval_steps": 200,
        "save_steps": 200,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "adam_epsilon": 1e-8,
        "load_best_model_at_end": True,
        "gradient_accumulation_steps": 1,
        "bf16": True,
        "bf16_full_eval": True,
    },
    "Position": {
        "output_dir": CHECKPOINT_DIR,
        "report_to": None,
        "evaluation_strategy": "steps",
        "num_train_epochs": 4,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "learning_rate": 2e-5,
        "weight_decay": 0.0001,
        "data_seed": 0,
        "seed": 0,
        "logging_steps": 200,
        "eval_steps": 200,
        "save_steps": 200,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "adam_epsilon": 1e-8,
        "load_best_model_at_end": True,
        "gradient_accumulation_steps": 1,
        "bf16": True,
        "bf16_full_eval": True,
    },
    "Evidence": {
        "output_dir": CHECKPOINT_DIR,
        "report_to": None,
        "evaluation_strategy": "steps",
        "num_train_epochs": 4,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "learning_rate": 2e-5,
        "weight_decay": 0.0001,
        "data_seed": 0,
        "seed": 0,
        "logging_steps": 200,
        "eval_steps": 200,
        "save_steps": 200,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "adam_epsilon": 1e-8,
        "load_best_model_at_end": True,
        "gradient_accumulation_steps": 1,
        "bf16": True,
        "bf16_full_eval": True,
    },
    "Lead": {
        "output_dir": CHECKPOINT_DIR,
        "report_to": None,
        "evaluation_strategy": "steps",
        "num_train_epochs": 4,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "learning_rate": 2e-5,
        "weight_decay": 0.0001,
        "data_seed": 0,
        "seed": 0,
        "logging_steps": 200,
        "eval_steps": 200,
        "save_steps": 200,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "adam_epsilon": 1e-8,
        "load_best_model_at_end": True,
        "gradient_accumulation_steps": 1,
        "bf16": True,
        "bf16_full_eval": True,
    },
    "Concluding Statement": {
        "output_dir": CHECKPOINT_DIR,
        "report_to": None,
        "evaluation_strategy": "steps",
        "num_train_epochs": 4,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "learning_rate": 2e-5,
        "weight_decay": 0.0001,
        "data_seed": 0,
        "seed": 0,
        "logging_steps": 200,
        "eval_steps": 200,
        "save_steps": 200,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "adam_epsilon": 1e-8,
        "load_best_model_at_end": True,
        "gradient_accumulation_steps": 1,
        "bf16": True,
        "bf16_full_eval": True,
    },
    "Counterclaim": {
        "output_dir": CHECKPOINT_DIR,
        "report_to": None,
        "evaluation_strategy": "steps",
        "num_train_epochs": 8,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "learning_rate": 2e-5,
        "weight_decay": 0.0001,
        "data_seed": 0,
        "seed": 0,
        "logging_steps": 200,
        "eval_steps": 200,
        "save_steps": 200,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "adam_epsilon": 1e-8,
        "load_best_model_at_end": True,
        "gradient_accumulation_steps": 1,
        "bf16": True,
        "bf16_full_eval": True,
    },
}

training_parameters_per_category = {}
for key, item in experimental_args.items():
    training_parameters_per_category[key] = TrainingArguments(**item)
