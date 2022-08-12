import os

FP_PREPROCESSED_TRAIN_CSV = "./intermediary_state/train.csv"
FP_PREPROCESSED_TEST_CSV = "./intermediary_state/test.csv"
FP_ORIGINAL_TRAIN_CSV = "./data/train.csv"
FP_ORIGINAL_TEST_CSV = "./data/test.csv"
FP_ORIGINAL_TRAIN_ESSAY_DIR = "./data/train/"
FP_ORIGINAL_TEST_ESSAY_DIR = "./data/test/"
FP_OUTPUT_SAMPLE_SUBMISSION_CSV = "./data/submission_fullset.csv"
FP_PREPROCESSED_BY_CATEGORY_CSV_DIR = "./intermediary_state/by_category"
FP_MERGED_FILE_OUTPUT = "./data/merged_file.txt"
FP_GENERATED_DIR = "./generated_content"

TRAINED_MODELS = {
    "bert-base-uncased-20test": "./trained_models/bert-uncased-20test",
    "bert-base-uncased-50test": "./trained_models/bert-uncased-50test",
    "bert-base-uncased": "./trained_models/bert-uncased",

    "microsoft/deberta-v3-large": "./trained_models/microsoft-deberta-v3-large",
    "microsoft/mdeberta-v3-base": "./trained_models/microsoft-mdeberta-v3-base",
    "microsoft/deberta-v3-xsmall": "./trained_models/microsoft-deberta-v3-xsmall",
    "gpt2": "./trained_models/gpt2-text-generation",
}

BY_CATEGORY_TRAINED_MODEL_DIR = "by_category"
MODEL_NAME_IN_USE = "bert-base-uncased"
EXPERIMENT_SUFFIX = ""

# MODEL_NAME_IN_USE = "microsoft/deberta-v3-large"
# EXPERIMENT_SUFFIX = ""

# MODEL_NAME_IN_USE = "microsoft/mdeberta-v3-base"
# EXPERIMENT_SUFFIX = ""

# MODEL_NAME_IN_USE = "microsoft/deberta-v3-xsmall"
# EXPERIMENT_SUFFIX = ""

# MODEL_NAME_IN_USE = "gpt2"
# EXPERIMENT_SUFFIX = ""

TRAINED_MODEL_KEY = MODEL_NAME_IN_USE + EXPERIMENT_SUFFIX

FP_TRAINED_MODEL_IN_USE = TRAINED_MODELS[TRAINED_MODEL_KEY]
CHECKPOINT_DIR = f"./checkpoint_{TRAINED_MODEL_KEY}/"

for dir_ in [FP_TRAINED_MODEL_IN_USE, CHECKPOINT_DIR]:
    try:
        os.makedirs(dir_)
    except FileExistsError:
        pass

TEST_SIZE = 0.2
NUM_LABELS = 3
TOKENIZER_MAX_SIZE = 512
USE_SMALL_DATASET = False
SPLIT_BY_CATEGORY = False

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
        save_steps=300,
        eval_steps=300,
        # todo:
        # try:
        # num_train_epochs=5,
        # save_steps = 1500,
        # eval_steps = 1500 
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
    ),  # {'eval_loss': 0.833855926990509, 'eval_accuracy': 0.6364575966925964, 'eval_runtime': 232.4464, 'eval_samples_per_second': 79.085, 'eval_steps_per_second': 4.943, 'epoch': 0.09}
    "bert-base-uncased": TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        report_to=None,
        evaluation_strategy="steps",
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        load_best_model_at_end=True,
        save_steps=500,
        eval_steps=500,
        learning_rate=0.000001,
        weight_decay=0.01,
        gradient_accumulation_steps=1
    ), 
    "bert-base-uncased-50test": TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        report_to=None,
        evaluation_strategy="steps",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        load_best_model_at_end=True,
        save_steps=150,
        eval_steps=150,
    ),  # {'eval_loss': 0.7443203926086426, 'eval_accuracy': 0.6699668171680356, 'eval_runtime': 146.1931, 'eval_samples_per_second': 125.745, 'eval_steps_per_second': 3.933, 'epoch': 0.78}
    "microsoft/deberta-v3-xsmall": TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        report_to=None,
        evaluation_strategy="steps",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        load_best_model_at_end=True,
        save_steps=300,
        eval_steps=300,
    ),  # {'eval_loss': 0.6639842987060547, 'eval_accuracy': 0.7019528912582277, 'eval_runtime': 103.2597, 'eval_samples_per_second': 178.027, 'eval_steps_per_second': 5.568, 'epoch': 1.57}
    "gpt2": TrainingArguments(
        output_dir=CHECKPOINT_DIR,  # The output directory
        num_train_epochs=5,  # number of training epochs
        per_device_train_batch_size=64,  # batch size for training
        # per_device_eval_batch_size=64,  # batch size for evaluation
        # eval_steps = 400, # Number of update steps between two evaluations.
        save_steps=2000,  # after # steps model is saved
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
    ),
}

training_parameters_per_category = {
    "Claim": TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        report_to=None,
        evaluation_strategy="steps",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        load_best_model_at_end=True,
        save_steps=20,
        eval_steps=20,
    ),  # Loading best model from ./checkpoint_microsoft/deberta-v3-xsmall/checkpoint-160 (score: 0.7820942401885986).
    "Concluding Statement": TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        report_to=None,
        evaluation_strategy="steps",
        num_train_epochs=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        load_best_model_at_end=True,
        save_steps=10,
        eval_steps=10,
    ),  # Loading best model from ./checkpoint_microsoft/deberta-v3-xsmall/checkpoint-40 (score: 0.6411638855934143).
    # Loading best model from ./checkpoint_bert-base-uncased/checkpoint-100 (score: 0.6916427612304688)
    "Counterclaim": TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        report_to=None,
        evaluation_strategy="steps",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        load_best_model_at_end=True,
        save_steps=20,
        eval_steps=20,
    ),  # Loading best model from ./checkpoint_microsoft/deberta-v3-xsmall/checkpoint-20 (score: 0.7619290947914124).
    # Loading best model from ./checkpoint_bert-base-uncased/checkpoint-40 (score: 0.6475394368171692)
    "Evidence": TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        report_to=None,
        evaluation_strategy="steps",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        load_best_model_at_end=True,
        save_steps=150,
        eval_steps=150,
    ),  # Loading best model from ./checkpoint_microsoft/deberta-v3-xsmall/checkpoint-150 (score: 0.7437736392021179).
    # Loading best model from ./checkpoint_bert-base-uncased/checkpoint-300 (score: 0.6900184750556946)
    "Lead": TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        report_to=None,
        evaluation_strategy="steps",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        load_best_model_at_end=True,
        save_steps=10,
        eval_steps=10,
    ),  # Loading best model from ./checkpoint_microsoft/deberta-v3-xsmall/checkpoint-70 (score: 0.7353153228759766).
    # Loading best model from ./checkpoint_bert-base-uncased/checkpoint-120 (score: 0.7210341691970825).
    "Position": TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        report_to=None,
        evaluation_strategy="steps",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        load_best_model_at_end=True,
        save_steps=20,
        eval_steps=20,
    ),  # Loading best model from ./checkpoint_microsoft/deberta-v3-xsmall/checkpoint-80 (score: 0.6482376456260681)
    # Loading best model from ./checkpoint_bert-base-uncased/checkpoint-60 (score: 0.6023277640342712)
    "Rebuttal": TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        report_to=None,
        evaluation_strategy="steps",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        load_best_model_at_end=True,
        save_steps=10,
        eval_steps=10,
    ),  # Loading best model from ./checkpoint_microsoft/deberta-v3-xsmall/checkpoint-30 (score: 0.8033574819564819)
    # Loading best model from ./checkpoint_bert-base-uncased/checkpoint-40 (score: 0.7233213782310486).
}

training_args = training_parameters[TRAINED_MODEL_KEY]
