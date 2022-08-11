from config import config
import pandas as pd

df = pd.read_csv(config.FP_ORIGINAL_TRAIN_CSV)
current_essay_id = ""

with open(config.FP_MERGED_FILE_OUTPUT, "w") as fp:
    for index, row in df.iterrows():
        essay_id = row.essay_id
        if essay_id != current_essay_id:
            # print("Changing essay ", essay_id)
            current_essay_id = essay_id
            fp.write(f"Essay: {current_essay_id}\n")
        fp.write(row.discourse_type)
        fp.write(" ")
        fp.write(row.discourse_effectiveness)
        fp.write(": \n")
        fp.write(row.discourse_text)
        fp.write("\n\n")
