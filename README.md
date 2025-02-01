# LLMS Know More Than They Show - extended Experimentation on Medical Domain

## Overview

This repository contains the codebase for the extended experiments based on the paper "LLMs Know More Than They Show" by Hadas et al. Our work focuses on analyzing hallucinations in Large Language Models (LLMs) within the medical domain. We utilized the MedQuAD dataset and extended the original code to work with the LLAMA 3.2 1B model, incorporating LoRA fine-tuning to enhance the model's medical knowledge.

> **Note:** This is an important note.

## Repository Structure

- `src/`: Contains the scource code of our experiments.
- `notebooks/` : Contains the notebooks we used for the experiments. 
- `README.md`: This file, providing an overview and instructions for the repository.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sriharshapy/LLM-knows-more-than-they-show.git
2. **Create a Virtual Environment**  
   ```bash
    python3 -m venv venv
    source venv/bin/activate
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt


# Fine-Tuning LLaMA 3.2 1B Model with MedQuAD Dataset
- `notebooks/llms_finetune_llama_1B.ipynb`
  
The provided code demonstrates the fine-tuning of the LLaMA 3.2 1B model using the MedQuAD dataset, which comprises over 47,000 medical question-and-answer pairs sourced from National Institutes of Health (NIH) websites. This process aims to enhance the model's proficiency in understanding and generating medical-related content.

## Training Process

1. **Environment Setup**:
   - Mount Google Drive to access and store data.
   - Install necessary libraries:
     - `accelerate`, `peft`, `transformers`, `trl`, `bitsandbytes` for model training and optimization.
     - `datasets` for handling the MedQuAD dataset.
     - Authenticate with Hugging Face to access pre-trained models and datasets.

2. **Data Preparation**:
   - Load the MedQuAD dataset.
   - Split the dataset into training and testing subsets.
   - Transform the data to align with the LLaMA model's input format, incorporating system prompts and structuring the conversation for effective training.

3. **Model Configuration**:
   - **Model Selection**: Utilize the `meta-llama/Llama-3.2-1B` model from the Hugging Face hub.
   - **LoRA Parameters**:
     - `lora_r = 64`: Defines the rank of the LoRA update matrices.
     - `lora_alpha = 16`: Scaling factor for the LoRA updates.
     - `lora_dropout = 0.1`: Dropout rate to prevent overfitting during training.
   - **Training Arguments**:
     - `output_dir`: Directory to save model checkpoints and outputs.
     - `num_train_epochs = 3`: Number of times the model will iterate over the entire training dataset.
     - `per_device_train_batch_size = 4`: Number of samples per batch for training.
     - `learning_rate = 2e-4`: Step size for the optimizer.
     - `weight_decay = 0.001`: Regularization parameter to prevent overfitting.
     - `optim = "paged_adamw_32bit"`: Optimizer choice for training.
     - `lr_scheduler_type = "cosine"`: Learning rate scheduler to adjust the learning rate during training.

4. **Training Execution**:
   - Initialize the `SFTTrainer` with the model, dataset, tokenizer, and training arguments.
   - Commence the training process, allowing the model to learn from the medical Q&A data.

5. **Model Evaluation and Deployment**:
   - After training, generate responses to medical queries to assess the model's performance.
   - Save and push the fine-tuned model and tokenizer to the Hugging Face hub for future use. Model is available at this repo [hitmanonholiday/LLAMA-3.2-1B-medical-qa](https://huggingface.co/hitmanonholiday/LLAMA-3.2-1B-medical-qa)



