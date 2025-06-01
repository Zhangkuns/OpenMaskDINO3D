from numba.core.ir import Print
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
llama_model_path = "llm/vicuna-7b-v1.5"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LlamaForCausalLM.from_pretrained(
    llama_model_path,
    torch_dtype=torch.bfloat16,
    #quantization_config=quantization_config,
    load_in_8bit=True,
    device_map="auto",
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=False, legacy=False)

prompt = "How many pairs of shoes is the table behind? Please output the related mask."
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

# Generate
generate_ids = model.generate(
    inputs.input_ids,
    max_length=100,
    do_sample=True,
    min_length=2,
    top_p=0.9,
    repetition_penalty=3.0,
    length_penalty=1,
    temperature=1.0,)
response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(f"Prompt: {prompt}")
print(f"Response: {response}")
