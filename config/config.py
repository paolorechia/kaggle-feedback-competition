import os
import torch

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


TRAINED_MODELS = {
    "bert-base-uncased-20test": "./trained_models/bert-uncased-20test",
    "bert-base-uncased-50test": "./trained_models/bert-uncased-50test",
    "bert-base-uncased": "./trained_models/bert-uncased",
    "bert-large-uncased": "./trained_models/bert-large-uncased",
    "microsoft/deberta-v3-large": "./trained_models/microsoft-deberta-v3-large",
    "microsoft/mdeberta-v3-base": "./trained_models/microsoft-mdeberta-v3-base",
    "microsoft/deberta-v3-base": "./trained_models/microsoft-deberta-v3-base",
    "microsoft/deberta-v3-xsmall": "./trained_models/microsoft-deberta-v3-xsmall",
    "microsoft/deberta-v2-xlarge-mnli": "./trained_models/microsoft/deberta-v2-xlarge-mnli",
    "gpt2": "./trained_models/gpt2-text-generation",
}

BY_CATEGORY_TRAINED_MODEL_DIR = "by_category"
# MODEL_NAME_IN_USE = "bert-base-uncased"
# EXPERIMENT_SUFFIX = ""
# Overfit test fullset: 50%
"""
{'accuracy': 0.625}
Claim {'accuracy': 0.59375}
Concluding Statement {'accuracy': 0.59375}
Counterclaim {'accuracy': 0.84375}
Evidence {'accuracy': 0.625}
Lead {'accuracy': 0.59375}
Position {'accuracy': 0.8125}
Rebuttal {'accuracy': 0.625}
"""


MODEL_NAME_IN_USE = "microsoft/deberta-v3-large"
EXPERIMENT_SUFFIX = ""
# Overfit test fullset: 100%
"""
{'accuracy': 1.0}
Claim {'accuracy': 1.0}
Concluding Statement {'accuracy': 1.0}
Counterclaim {'accuracy': 1.0}
Evidence {'accuracy': 1.0}
Lead {'accuracy': 1.0}
Position {'accuracy': 1.0}
Rebuttal {'accuracy': 1.0}
"""

# MODEL_NAME_IN_USE = "microsoft/deberta-v2-xlarge-mnli"
# EXPERIMENT_SUFFIX = ""
# Overfit test fullset: 100%

# MODEL_NAME_IN_USE = "microsoft/mdeberta-v3-base"
# EXPERIMENT_SUFFIX = ""
# Overfit test fullset: 68%

# MODEL_NAME_IN_USE = "microsoft/deberta-v3-base"
# EXPERIMENT_SUFFIX = ""
# Overfit test fullset: 62%
"""
Claim {'accuracy': 0.625}
Concluding Statement {'accuracy': 0.5625}
Counterclaim {'accuracy': 0.9375}
Evidence {'accuracy': 0.75}
Lead {'accuracy': 0.8125}
Position {'accuracy': 0.8125}
Rebuttal {'accuracy': 0.625}
"""

# MODEL_NAME_IN_USE = "microsoft/deberta-v3-xsmall"
# EXPERIMENT_SUFFIX = ""
# Overfit test fullset: 50%
"""
# Overfit test by category
Claim {'accuracy': 0.5}
Concluding Statement {'accuracy': 0.5625}
Counterclaim {'accuracy': 0.84375}
Evidence {'accuracy': 0.625}
Lead {'accuracy': 0.71875}
Position {'accuracy': 0.8125}
Rebuttal {'accuracy': 0.5625}
"""

# MODEL_NAME_IN_USE = "gpt2"
# EXPERIMENT_SUFFIX = ""

TRAINED_MODEL_KEY = MODEL_NAME_IN_USE + EXPERIMENT_SUFFIX

FP_TRAINED_MODEL_IN_USE = TRAINED_MODELS[TRAINED_MODEL_KEY]
CHECKPOINT_DIR = f"./checkpoint_{TRAINED_MODEL_KEY}/"

for dir_ in [FP_TRAINED_MODEL_IN_USE, CHECKPOINT_DIR, FP_EXPERIMENT_PARAMETERS_DIR]:
    try:
        os.makedirs(dir_)
    except FileExistsError:
        pass

TEST_SIZE = 0.2
VAL_SIZE = 0.223
NUM_LABELS = 3
TOKENIZER_MAX_SIZE = 1024
SPLIT_BY_CATEGORY = False
HIDDEN_DROPOUT_PROB = 0.3
ATTENTION_PROBS_DROPOUT_PROB = 0.3
LAYER_NORM_EPS = 1e-7

METRIC = "accuracy"

from transformers import DebertaV2Config, DebertaV2Model

def override_deberta_v2_config(model: DebertaV2Model):
    deberta_config: DebertaV2Config = model.config
    deberta_config.max_position_embeddings = TOKENIZER_MAX_SIZE
    deberta_config.hidden_dropout_prob = HIDDEN_DROPOUT_PROB
    deberta_config.attention_probs_dropout_prob = ATTENTION_PROBS_DROPOUT_PROB
    deberta_config.layer_norm_eps = LAYER_NORM_EPS
    return deberta_config

from transformers import TrainingArguments

training_parameters = {}

experimental_args = {
    "Rebuttal": {
        "output_dir": CHECKPOINT_DIR,
        "report_to": None,
        "evaluation_strategy": "epoch",
        "num_train_epochs": 8,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "learning_rate": 0.0001,
        "weight_decay": 1,
        "data_seed": 0,
        "seed": 0,
        "logging_steps": 10,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "adam_epsilon": 1e-5,
        "gradient_accumulation_steps": 10,
        "bf16": True,
        "bf16_full_eval": True,
    },
    "Claim": {
        "output_dir": CHECKPOINT_DIR,
        "report_to": None,
        "evaluation_strategy": "epoch",
        "num_train_epochs": 2,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "learning_rate": 0.0001,
        "weight_decay": 1,
        "data_seed": 0,
        "seed": 0,
        "logging_steps": 10,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "adam_epsilon": 1e-5,
        "gradient_accumulation_steps": 10,
        "bf16": True,
        "bf16_full_eval": True,
    },
}

training_parameters_per_category = {
    "Claim": TrainingArguments(**experimental_args["Claim"]),
    "Concluding Statement": {},
    "Counterclaim": {},
    "Evidence": {},
    "Lead": {},
    "Position": {},
    "Rebuttal": TrainingArguments(**experimental_args["Rebuttal"])
}

training_args = training_parameters[TRAINED_MODEL_KEY]
