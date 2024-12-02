from datasets import load_dataset

print("Starting dataset load...")
dataset = load_dataset("locuslab/TOFU", "forget01", split="train")
print("Dataset loaded successfully.")
