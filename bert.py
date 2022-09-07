import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from config import *


def process_dataset(train_path, test_path, label_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_train.rename(columns={'content': 'text'}, inplace=True)
    df_test.rename(columns={'content': 'text'}, inplace=True)

    label_dict = np.load(label_path, allow_pickle=True).tolist()
    label_dict_reverse = {}
    for key, val in label_dict.items():
        label_dict_reverse[val] = int(key)
    df_train["label"] = df_train.label.map(label_dict_reverse)
    df_test["label"] = df_test.label.map(label_dict_reverse)

    folder_path = os.path.abspath("./process_data")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    csv_train_path = os.path.join(folder_path, "train.csv")
    csv_test_path = os.path.join(folder_path, "test.csv")
    df_train.to_csv(csv_train_path, index=False)
    df_test.to_csv(csv_test_path, index=False)
    print("data process finished.")

    return csv_train_path, csv_test_path


def finetune(train_path, test_path, label_path, pretrained_model, max_iteration, cache_dir):

    csv_train_path, csv_test_path = process_dataset(
        train_path, test_path, label_path)
    data = load_dataset('csv', data_files={
        'train': csv_train_path, 'test': csv_test_path}, cache_dir=cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model, cache_dir=cache_dir)
    tokenizer.model_max_length = 512
    tokenizer.max_len_single_sentence = 510

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_data = data.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model, num_labels=236, cache_dir=cache_dir)

    per_device_batch_size = 4

    training_args = TrainingArguments(
        output_dir="./result",
        learning_rate=1e-5,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        weight_decay=0.01
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.create_optimizer()

    num_iter, epoch = 0, 0
    while True:
        data_loader = trainer.get_train_dataloader()
        for batch in tqdm(data_loader):
            trainer.optimizer.zero_grad()
            trainer.training_step(trainer.model, batch)
            trainer.optimizer.step()

            num_iter += 1
        print('epoch {} finished'.format(epoch))
        trainer.save_model('./finetuned/')

        pred = trainer.predict(tokenized_data['test']).predictions
        pred_prob = np.exp(pred - np.max(pred, axis=1, keepdims=True)) / \
            np.sum(np.exp(pred - np.max(pred, axis=1, keepdims=True)),
                   axis=1, keepdims=True)
        performance = evaluation(pred_prob, tokenized_data['test']['label'])
        print(performance)
        epoch += 1
        if num_iter >= max_iteration:
            break
    print('finished')


def evaluation(predict_prob, label):
    predict = np.argmax(predict_prob, axis=1)
    micro_recall = recall_score(label, predict, average='micro')
    macro_recall = recall_score(label, predict, average='macro')
    micro_precision = precision_score(label, predict, average='micro')
    macro_precision = precision_score(label, predict, average='macro')
    micro_f1 = f1_score(label, predict, average='micro')
    macro_f1 = f1_score(label, predict, average='macro')
    accuracy = accuracy_score(label, predict)

    # micro_auc = roc_auc_score(label, predict_prob, multi_class='ovo')
    # macro_auc = roc_auc_score(label, predict_prob, multi_class='ovr')
    performance = {
        'accuracy': accuracy,
        'micro_recall': micro_recall,
        'macro_recall': macro_recall,
        'micro_precision': micro_precision,
        'macro_precision': macro_precision,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        # 'micro_auc': micro_auc,
        # 'macro_auc': macro_auc,
    }
    return performance


if __name__ == '__main__':
    finetune(train_path, test_path, label_path, pretrained_model,
             max_iteration, cache_dir)


# pred = trainer.predict(tokenized_data['test'][:10]).predictions
# pred_prob = np.exp(pred - np.max(pred, axis=1, keepdims=True)) / \
#     np.sum(np.exp(pred - np.max(pred, axis=1, keepdims=True)),
#            axis=1, keepdims=True)
# performance = evaluation(pred_prob, tokenized_data['test']['label'])
# pd.DataFrame(performance, index=0).to_csv(
#     "/mnt/hdd3/shihanrui/bert/result/performance.csv", mode='a')
