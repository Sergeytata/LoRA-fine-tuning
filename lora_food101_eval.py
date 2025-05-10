from peft import PeftConfig, PeftModel
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
import argparse

from PIL import Image
import requests

import matplotlib.pyplot as plt 

import json

def main(args):
    repo_name = args.repo_name

    with open(f"lora_food101_id2label.json", "r") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    with open(f"lora_food101_label2id.json", "r") as f:
        label2id = json.load(f)
    
    config = PeftConfig.from_pretrained(repo_name)

    model = AutoModelForImageClassification.from_pretrained(
        config.base_model_name_or_path,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )
    # Load the LoRA model
    inference_model = PeftModel.from_pretrained(model, repo_name)


    url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)
    

    image_processor = AutoImageProcessor.from_pretrained(repo_name)


    encoding = image_processor(image.convert("RGB"), return_tensors="pt")


    with torch.no_grad():
        outputs = inference_model(**encoding)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", inference_model.config.id2label[predicted_class_idx])

    # Display the image
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted class: {inference_model.config.id2label[predicted_class_idx]}")
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Food101 Evaluation")
    parser.add_argument(
        "--repo_name",
        type=str,
        default="sayakpaul/lora-food101",
        help="The name of the repository to eval the model.",
    )
    args = parser.parse_args()

    main(args)