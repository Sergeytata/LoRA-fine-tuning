import torch
# torch.autograd.set_detect_anomaly(True)

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# Load base model and tokenizer
model_name = "tiiuae/falcon-7b"  # or another suitable model under 24GB VRAM
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    quantization_config=bnb_config,
)
model.config.use_cache = False

# Apply LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
lora_model = get_peft_model(model, lora_config)

# Load training data
dataset = load_dataset("Abirate/english_quotes")  # example dataset

tokenizer.pad_token = tokenizer.eos_token
tokenized = dataset.map(lambda e: tokenizer(e['quote'], truncation=True, padding="max_length", max_length=128), batched=True)
tokenized["train"] = tokenized["train"].map(
    lambda example: {"labels": example["input_ids"]},
    remove_columns=[],  # Optional: preserves all other fields
)


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


print_trainable_parameters(lora_model)



# Training
# TODO: Add evaluation and metrics for this case
# TODO: Consider other datasets for training
# TODO: Find a way to to know exactly how model is split across GPUs
# TODO: Add arguments on how to split batch size between GPUs
training_args = TrainingArguments(
    output_dir=f"models/faclon-7b-finetuned-lora-english-quotes",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=1,
    fp16=True,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    report_to="none"
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized["train"]
)


trainer.train()