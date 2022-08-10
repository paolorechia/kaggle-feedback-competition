import os

FP_PREPROCESSED_TRAIN_CSV = "./intermediary_state/train.csv"
FP_PREPROCESSED_TEST_CSV = "./intermediary_state/test.csv"
FP_ORIGINAL_TRAIN_CSV = "./data/train.csv"
FP_ORIGINAL_TEST_CSV = "./data/test.csv"
FP_OUTPUT_SAMPLE_SUBMISSION_CSV = "./data/submission_fullset.csv"

TRAINED_MODELS = {
    "bert-base-uncased-20test": "./trained_models/bert-uncased-20test",
    "bert-base-uncased-50test": "./trained_models/bert-uncased-50test",
    "microsoft/deberta-v3-large": "./trained_models/microsoft-deberta-v3-large",
    "microsoft/mdeberta-v3-base": "./trained_models/microsoft-mdeberta-v3-base",
    "microsoft/deberta-v3-xsmall": "./trained_models/microsoft-deberta-v3-xsmall"
}

# MODEL_NAME_IN_USE = "bert-base-uncased"
# EXPERIMENT_SUFFIX = "-50test"

# MODEL_NAME_IN_USE = "microsoft/deberta-v3-large"
# EXPERIMENT_SUFFIX = ""

# MODEL_NAME_IN_USE = "microsoft/mdeberta-v3-base"
# EXPERIMENT_SUFFIX = ""

MODEL_NAME_IN_USE = "microsoft/deberta-v3-xsmall"
EXPERIMENT_SUFFIX = ""

TRAINED_MODEL_KEY = MODEL_NAME_IN_USE + EXPERIMENT_SUFFIX

FP_TRAINED_MODEL_IN_USE = TRAINED_MODELS[TRAINED_MODEL_KEY]
CHECKPOINT_DIR = f"./checkpoint_{TRAINED_MODEL_KEY}/"

for dir_ in [FP_TRAINED_MODEL_IN_USE, CHECKPOINT_DIR]:
    try:
        os.makedirs(dir_)
    except FileExistsError:
        pass

TEST_SIZE = 0.5
NUM_LABELS = 3
TOKENIZER_MAX_SIZE = 512

USE_SMALL_DATASET = False

METRIC = "accuracy"

from transformers import TrainingArguments

training_parameters = {
    "microsoft/deberta-v3-large": TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        report_to=None,
        evaluation_strategy="steps",
        num_train_epochs=1,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        load_best_model_at_end=True,
        save_steps=1000,
        eval_steps=1000,
    ),
    "microsoft/mdeberta-v3-base": TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        report_to=None,
        evaluation_strategy="steps",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        load_best_model_at_end=True,
        save_steps=100,
        eval_steps=100,
    ), # {'eval_loss': 0.833855926990509, 'eval_accuracy': 0.6364575966925964, 'eval_runtime': 232.4464, 'eval_samples_per_second': 79.085, 'eval_steps_per_second': 4.943, 'epoch': 0.09}
    "bert-base-uncased-50test": TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        report_to=None,
        evaluation_strategy="steps",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        load_best_model_at_end=True,
        save_steps=150,
        eval_steps=150,
    ), # {'eval_loss': 0.7443203926086426, 'eval_accuracy': 0.6699668171680356, 'eval_runtime': 146.1931, 'eval_samples_per_second': 125.745, 'eval_steps_per_second': 3.933, 'epoch': 0.78}
    "microsoft/deberta-v3-xsmall": TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        report_to=None,
        evaluation_strategy="steps",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        load_best_model_at_end=True,
        save_steps=150,
        eval_steps=150,
    ), # {'eval_loss': 0.7446180582046509, 'eval_accuracy': 0.6732307022792797, 'eval_runtime': 96.6335, 'eval_samples_per_second': 190.234, 'eval_steps_per_second': 5.95, 'epoch': 1.3}
}

training_args = training_parameters[TRAINED_MODEL_KEY]
