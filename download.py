from huggingface_hub import snapshot_download
import getpass
import os

# snapshot_download(repo_id=config.model.name, cache_dir=cache_dir)
# snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2')

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def download_data(subset, save_path):
    """
    Download a specific subset of a Hugging Face dataset and save it to disk.
    """
    print(f"Downloading subset '{subset}'...")
    try:
        # Load the specified subset of the dataset
        dataset = load_dataset("locuslab/TOFU", subset, split="train")

        cache_dir="/state/partition1/user/" + getpass.getuser() + "/hug"
        print(os.path.join(cache_dir, save_path))
        # Save the subset to the specified path
        dataset.save_to_disk(os.path.join(cache_dir, save_path))

        print(f"subset '{subset}' saved successfully at '{save_path}'!")
    except Exception as e:
        print(f"Error downloading subset: {e}")

if __name__ == "__main__":

    # Call the download_subset function with the parsed arguments

    # retain_subset = "retain" + str(100 - int(args.subset.replace("forget", ""))).zfill(2)

    # for subset, path in [("retain90", "./retain90_data")]:
    # ,
    #                      ("retain99", "./retain_data"),
    #                      ("retain_perturbed", "./retain_perturbed_data"),
    #                      ("real_authors_perturbed", "./real_authors_perturbed_data"),
    #                      ("world_facts_perturbed", "./world_facts_perturbed_data")]:
        # download_data(subset, path)


    model_name = "locuslab/tofu_ft_llama2-7b"  # Hugging Face model path for LLaMA 2 7B
    # meta-llama/Llama-2-7b-chat-hf
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # model_name = "openai-community/gpt2-xl"
    cache_dir="/state/partition1/user/" + getpass.getuser() + "/hug"
    print(f'downloading {model_name}')
    snapshot_dir = snapshot_download(repo_id=model_name, cache_dir=cache_dir)
    print(f'snapshot: {snapshot_dir}')
    # for model_name in ["locuslab/llama2-7b_idk_1e-05_forget10", 'locuslab/llama2-7b_grad_diff_1e-05_forget10', 'locuslab/llama2-7b_grad_ascent_1e-05_forget10', 'locuslab/llama2-7b_KL_1e-05_forget10']:
    #     print(f'downloading {model_name}')
    #     checkpoint = 'checkpoint-60'
    #     snapshot_dir = snapshot_download(repo_id=model_name, revision=checkpoint, cache_dir=f'{cache_dir}/{checkpoint})
    #     print(f'snapshot: {snapshot_dir}')