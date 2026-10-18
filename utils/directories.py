PRETRAINING_PAGES_DIR = "data/pages.json"
PRETRAINING_RAW_DATASET_DIR = "data/datasets/text.txt"
PRETRAINING_BIN_DATASET_DIR = "data/datasets/train.bin"
PRETRAINING_BIN_VALIDATION_DATASET_DIR = "data/datasets/validate.bin"
PRETRAINING_MERGES_DIR = "data/merges.json"

SFT_RAW_DATASET_DIR = "data/datasets/sft/text.txt"
SFT_BIN_DATASET_DIR = "data/datasets/sft/train.bin"
SFT_BIN_VALIDATION_DATASET_DIR = "data/datasets/sft/validate.bin"
# sft has to use the same tokenizer the model was pretrained with
SFT_MERGES_DIR = PRETRAINING_MERGES_DIR
