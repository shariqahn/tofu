from huggingface_hub import snapshot_download
import getpass
import os

# snapshot_download(repo_id=config.model.name, cache_dir=cache_dir)
# snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2')

from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from datasets import load_dataset, get_dataset_config_names

def download_data(subset, save_path):
    """
    Download a specific subset of a Hugging Face dataset and save it to disk.
    """
    print(f"Downloading subset '{subset}'...")
    try:
        # Load the specified subset of the dataset
        data_path = "locuslab/TOFU"
        # configs = get_dataset_config_names(data_path)
        # print("Available configurations:", configs)
        dataset = load_dataset("locuslab/TOFU","forget01", split="train")
        print(dataset[0])

        cache_dir="/state/partition1/user/" + getpass.getuser() + "/hug"
        print(os.path.join(cache_dir, save_path))
        # Save the subset to the specified path
        dataset.save_to_disk(os.path.join(cache_dir, save_path))

        print(f"subset '{subset}' saved successfully at '{save_path}'!")
    except Exception as e:
        print(f"Error downloading subset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subset", type=str)
    parser.add_argument("forget_path", type=str)
    parser.add_argument("retain_path", type=str)

    args = parser.parse_args()

    # Call the download_subset function with the parsed arguments
    download_data(args.subset, args.forget_path)

    retain_subset = "retain" + str(100 - int(args.subset.replace("forget", ""))).zfill(2)
    download_data(retain_subset, args.retain_path)


    # model_name = "meta-llama/Llama-2-7b-hf"  # Hugging Face model path for LLaMA 2 7B
    # # meta-llama/Llama-2-7b-chat-hf
    # # model_name = "openai-community/gpt2-xl"
    # cache_dir="/state/partition1/user/" + getpass.getuser() + "/hug"
    # print(f'downloading {model_name}')
    # snapshot_dir = snapshot_download(repo_id=model_name, cache_dir=cache_dir)
    # print(f'snapshot: {snapshot_dir}')
    # # pdb.set_trace()

    # # model = AutoModelForCausalLM.from_pretrained(snapshot_dir, device_map=None, local_files_only=True, torch_dtype=torch.float32)
    # # tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)

    # # # # Define the model name and path
    # # # model_name = '/home/gridsan/shossain/EasyEdit/scr/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9'

    # # # # Load the model and tokenizer using the transformers library
    # # # model = AutoModelForCausalLM.from_pretrained(model_name, device_map=None, torch_dtype=torch.float32, local_files_only=True)
    # # # tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    # # # pdb.set_trace()

    # # # Example usage
    # # input_text = "Where is the eiffel tower?"
    # # inputs = tokenizer(input_text, return_tensors="pt")
    # # inputs = inputs.to('cuda')
    # # outputs = model.generate(inputs['input_ids'], max_length=50)
    # # print(tokenizer.decode(outputs[0], skip_special_tokens=True))

