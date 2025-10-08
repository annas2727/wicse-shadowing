import os
import torch
import librosa
import numpy as np
import pandas as pd
from datasets import Dataset, Audio, Features, ClassLabel
from transformers import (
    ASTFeatureExtractor,
    ASTConfig,
    ASTForAudioClassification,
    TrainingArguments,
    Trainer
)
import evaluate

audio_files = [
    r"C:/Users/annas/OneDrive/Documents/wicseSP/src/tests/rifle-gun.mp3",
    r"C:/Users/annas/OneDrive/Documents/wicseSP/src/tests/game-explosion.mp3"
]
labels = [0, 1]
label_names = ["gunfire", "explosion"]

df = pd.DataFrame({"audio": audio_files, "labels": labels})

features = Features({
    "audio": Audio(sampling_rate=16000),
    "labels": ClassLabel(names=label_names)
})
dataset = Dataset.from_pandas(df, features=features)

dataset = dataset.train_test_split(test_size=0.5)

pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)
config = ASTConfig.from_pretrained(pretrained_model)
config.num_labels = len(label_names)
config.id2label = {i: n for i, n in enumerate(label_names)}
config.label2id = {n: i for i, n in enumerate(label_names)}
model = ASTForAudioClassification.from_pretrained(
    pretrained_model, config=config, ignore_mismatched_sizes=True
)


SAMPLING_RATE = feature_extractor.sampling_rate
model_input_name = feature_extractor.model_input_names[0]

def preprocess(batch):
    audio_arrays = [a["array"] for a in batch["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt"
    )
    batch[model_input_name] = inputs[model_input_name]
    return batch

dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
dataset.set_transform(preprocess, output_all_columns=False)


accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=preds, references=eval_pred.label_ids)

training_args = TrainingArguments(
    output_dir="./runs/ast_demo",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    do_eval=True,
    save_strategy="epoch",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
print ("Training complete!")
