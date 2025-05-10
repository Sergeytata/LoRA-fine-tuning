# Author: Sergei Chirkunov
# LORA fine-tuning of ViT on Food101 dataset
# Adapted from the tutoiral https://huggingface.co/docs/peft/main/en/task_guides/image_classification_lora

import argparse

import transformers
import accelerate
import peft


# from huggingface_hub import notebook_login
# notebook_login()
# from huggingface_hub import login


from transformers import AutoImageProcessor
from datasets import load_dataset

# Preprocessing imports
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

# Model imports
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

# LORA imports
from peft import LoraConfig, get_peft_model

# Evaluation metrics
import numpy as np
import evaluate



import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


# Helper functions
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def main(args):
    username = args.username
    model_checkpoint = args.model_checkpoint
    # model_checkpoint = "google/vit-base-patch16-224-in21k"

    dataset = load_dataset("food101", split="train[:5000]")


    labels = dataset.features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    

    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(image_processor.size["height"]),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    val_transforms = Compose(
        [
            Resize(image_processor.size["height"]),
            CenterCrop(image_processor.size["height"]),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch


    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    splits = dataset.train_test_split(test_size=0.1)
    train_ds = splits["train"]
    val_ds = splits["test"]

    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)

    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )

    print_trainable_parameters(model)

    

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )
    lora_model = get_peft_model(model, config)

    print_trainable_parameters(lora_model)

    from transformers import TrainingArguments, Trainer


    model_name = model_checkpoint.split("/")[-1]
    batch_size = 128

    train_args = TrainingArguments(
        output_dir=f"models/{model_name}-finetuned-lora-food101",
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-3,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        num_train_epochs=5,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
        label_names=["labels"],
    )

    metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    # Train the model
    trainer = Trainer(
        lora_model,
        train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    train_results = trainer.train()
    print(train_results)


    # Evaluate the model
    trainer.evaluate(val_ds)

    if username is not None:
        repo_name = f"{username}/{model_name}-finetuned-lora-food101"
        lora_model.push_to_hub(repo_name)





if __name__ == "__main__":
    print(f"Transformers version: {transformers.__version__}")
    print(f"Accelerate version: {accelerate.__version__}")
    print(f"PEFT version: {peft.__version__}")



    parser = argparse.ArgumentParser(description="LORA fine-tuning of ViT on Food101 dataset")
    parser.add_argument("--username", type=str, required=False, default=None, help="Hugging Face username")
    parser.add_argument("--model_checkpoint", type=str, default="google/vit-base-patch16-224-in21k", help="Model checkpoint to use")
    args = parser.parse_args()

    if args.username is not None:
        import os
        # check if HF_TOKEN is set
        if "HF_TOKEN" not in os.environ:
            raise ValueError("Please set the HF_TOKEN environment variable with your Hugging Face token.")
        
    main(args)