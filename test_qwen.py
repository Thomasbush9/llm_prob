import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = 'Qwen/Qwen2.5-3B'
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map='cuda')
print('device:', model.device, '| dtype:', next(model.parameters()).dtype)

prompt = 'The capital of France is'
ids = tok(prompt, return_tensors='pt').to(model.device)
out = model.generate(**ids, max_new_tokens=20, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))

