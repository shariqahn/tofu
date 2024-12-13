# import random
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from datasets import load_from_disk

# # Define model paths
# llama = "/home/gridsan/shossain/tofu/scr/models--locuslab--tofu_ft_llama2-7b/snapshots/8fa500e8f345f1dd9cfe95bb4689878c944c9cbd"
# model_paths = [
#     # "/home/gridsan/shossain/tofu/scr/models--locuslab--tofu_ft_llama2-7b/snapshots/",
#     "/home/gridsan/shossain/EasyEdit/outputs/ROME_dummy/model",
#     llama
# ]

# # Define data path
# data_path = "./scr/forget10_perturbed_data"

# # Load data
# data = load_from_disk(data_path)

# # Define prompts column
# prompts_column = "question"
# random_prompts = random.sample(data[prompts_column], 5)

# # Define model and tokenizer
# model_name = llama  # replace with the actual model name
# tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

# for model_path in model_paths:
#     # Load model
#     model = AutoModelForCausalLM.from_pretrained(model_path)

#     # Tokenize prompts
#     inputs = tokenizer(random_prompts, return_tensors="pt", padding=True)
#     print('got inputs')
#     # Generate responses
#     outputs = model.generate(**inputs)
#     print('generated')
#     # Print responses
#     for i, output in enumerate(outputs):
#         print(f"Prompt: {random_prompts[i]}")
#         print(f"Response: {tokenizer.decode(output, skip_special_tokens=True)}")
#         print()

exec("""
for i in range(len(strs)):
    if strs[i] == '':
        print[i]
""")