import datasets
import random
import transformers
from transformers import AutoTokenizer

training_config = None
tokenizer = None

def get_tokenize_function(tokenizer, max_length):
    def tokenizer_function(examples):
        if 'question' in examples and 'answer' in examples:
            text = examples["question"][0] + examples['answer'][0]
        elif 'input' in examples and 'output' in examples:
            text = examples["input"][0] + examples['output'][0]
        else:
            text = examples["text"][0]

        tokenizer.pad_token = tokenizer.eos_token
        tokenized_inputs = tokenizer(text, 
                                    return_tensors='np', 
                                    truncation=True)
        mx_length = min(2048,
                        tokenized_inputs['input_ids'].shape[1])
        tokenizer.truncation_side = 'left'
        tokenized_inputs = tokenizer(text, 
                                    return_tensors='np', 
                                    truncation=True,
                                    max_length=mx_length)
        tokenized_inputs['labels'] = tokenized_inputs['input_ids'].copy()
        return tokenized_inputs
    return tokenizer_function

def load_dataset(dataset_path, tokenizer):
    random.seed(99)
    finetuning_dataset_loaded = datasets.load_dataset("json", data_files=dataset_path, split="train")
    tokenizer.pad_token = tokenizer.eos_token
    max_length = training_config["model"]["mx_length"]
    tokenized_dataset = finetuning_dataset_loaded.map(
        get_tokenize_function(tokenizer, max_length), # returns tokenize_function
        batched=True,
        batch_size=1,
        drop_last_batch=True
    )
    tokenized_dataset = tokenized_dataset.with_format("torch")
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=123)
    return split_dataset

def tokenize_and_split_data(config, tokenizer_):
    global training_config
    global tokenizer
    training_config = config
    tokenizer = tokenizer_
    dataset_path = config['datasets']['path']
    use_hf = config['datasets']['use_hf']
    if use_hf:
        finetuning_dataset_loaded = datasets.load_dataset('lamini/lamini_docs')
    else:
        finetuning_dataset_loaded = load_dataset(dataset_path, tokenizer)
    return finetuning_dataset_loaded['train'], finetuning_dataset_loaded['test']



# tokenized_dataset = finetuning_dataset_loaded.map(tokenizer_function,
#                                               batched=True,
#                                               batch_size=1,
#                                               drop_last_batch=True)
# tokenized_dataset = tokenized_dataset.add_column('labels', tokenized_dataset['input_ids'])
# split_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=123)
