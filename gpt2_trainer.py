from config import config

from transformers import AutoTokenizer

print(f"Loading model... {config.TRAINED_MODEL_KEY}")

tokenizer = AutoTokenizer.from_pretrained(config.TRAINED_MODEL_KEY)
from transformers import TextDataset, DataCollatorForLanguageModeling


def load_dataset(train_path, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer, file_path=train_path, block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    return train_dataset, data_collator


train_dataset, data_collator = load_dataset(config.FP_MERGED_FILE_OUTPUT, tokenizer)

from transformers import Trainer, AutoModelWithLMHead

model = AutoModelWithLMHead.from_pretrained(config.TRAINED_MODEL_KEY)

trainer = Trainer(
    model=model,
    args=config.training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

print(f"Will train and save results to... {config.FP_TRAINED_MODEL_IN_USE}")

trainer.train()

trainer.save_model(config.FP_TRAINED_MODEL_IN_USE)

print("Done!")
