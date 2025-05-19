import keras
print(keras.__version__)

import tf_keras;
print(tf_keras.__version__)

# tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-2-Pro-Mistral-7B")
# model = AutoModelForCausalLM.from_pretrained(
#     "NousResearch/Hermes-2-Pro-Mistral-7B",
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# LLM модель: Phi-3-mini
# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/Phi-3-mini-4k-instruct",
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# LLM модель (замените на вашу)
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct")