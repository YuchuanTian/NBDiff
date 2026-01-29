# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import types
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from generation_utils import diffusion_generate

# 1. Configuration and Model Loading
model_id = "yuchuantian/NBDiff-7B-Instruct"

print(f"Fetching model from Hugging Face: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    use_fast=False, 
    trust_remote_code=True,
    local_files_only=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",  # Automatically uses CUDA/GPU
    local_files_only=False
)

# Inject the custom diffusion generation method
model.diffusion_generate = types.MethodType(diffusion_generate, model)

mask_token_id = 45830
eos_token_id = tokenizer.eos_token_id

print("\n" + "="*50)
print("Model loaded successfully!")
print("Type 'exit' or 'quit' to stop the session.")
print("="*50)

# 2. Interactive Loop
while True:
    # Get user input from the terminal
    user_prompt = input("\nUser: ").strip()
    
    # Check for exit commands
    if not user_prompt:
        continue
    if user_prompt.lower() in ["exit", "quit"]:
        print("Closing session. Goodbye!")
        break

    # Construct the message structure for the chat template
    messages = [[{"role": "user", "content": user_prompt}]]
    
    # Apply the chat template specific to the model's Instruct format
    user_input = [tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
    ) for message in messages]

    # Encode the inputs and move to the model's device (CUDA)
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, padding_side="left")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # 3. Inference using the injected diffusion_generate method
    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            top_p=0.8,
            block_length=32,
            attention_mask=attention_mask,
            temperature=1.0,
            max_new_tokens=128,
            alg="entropy",
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            num_small_blocks=8
        )

    # 4. Decoding and Post-processing
    # Slice the output to get only the newly generated tokens
    generation = tokenizer.batch_decode(output[:, input_ids.shape[1]:].tolist())
    
    # Clean up the response by removing the EOS token and extra whitespace
    response = generation[0].split(tokenizer.eos_token)[0].strip()
    
    print(f"Assistant: {response}")