import os

from openai import OpenAI
from tqdm import tqdm

with open("/zhome/a8/4/207573/OpenAI-Keys/openai.txt", "r") as key_file:
    openai_key = key_file.read().strip()

client = OpenAI(api_key=openai_key)


def augment_with_gpt(prompt, human_like):
    # pipeline = transformers.pipeline("text-generation", model="meta-llama/Llama-3.2-1B", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto", use_auth_token="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")

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "instructions: paraphrase the input, to have the same meaning, but make it as natural for humans to write. Provide 5 possible outputs, avoid enumerating.",
            },
            {"role": "user", "content": human_like},
        ],
    )

    # remove blank lines
    my_string = completion.choices[0].message.content
    my_string = "\n".join([line for line in my_string.splitlines() if line.strip()])

    return my_string


def augment_prompt(dataset_path):
    # Loop through all the sample directories
    for sample_dir in tqdm(os.listdir(dataset_path), desc="Samples"):
        sample_path = os.path.join(dataset_path, sample_dir)

        # Check if the directory exists and is not a file
        if os.path.isdir(sample_path):
            # Loop through all the files in the sample directory
            for file in os.listdir(sample_path):
                if (
                    file.startswith("prompt_")
                    and file.endswith(".txt")
                    and "_gpt" not in file
                ):
                    # Extract the prompt index from the filename
                    prompt_idx = file.split("_")[-1].split(".")[0]

                    prompt_file = os.path.join(sample_path, f"prompt_{prompt_idx}.txt")
                    human_like_file = os.path.join(
                        sample_path, f"prompt_human_like_{prompt_idx}.txt"
                    )
                    gpt_prompt_file = os.path.join(
                        sample_path, f"prompt_gpt_{prompt_idx}.txt"
                    )

                    # Check if both files exist before proceeding
                    if os.path.exists(prompt_file) and os.path.exists(human_like_file):
                        with open(prompt_file, "r") as pf, open(
                            human_like_file, "r"
                        ) as hf:
                            prompt_content = pf.read().strip()
                            human_like_content = hf.read().strip()

                        # Combine the contents with the "bla bla" prefix
                        combined_content = augment_with_gpt(
                            prompt_content, human_like_content
                        )

                        # Write the combined content to the new file
                        with open(gpt_prompt_file, "w") as gf:
                            gf.write(combined_content)

                        print(f"Created: {gpt_prompt_file}")
                    else:
                        print(
                            f"Skipping: {prompt_file} or {human_like_file} not found."
                        )


if __name__ == "__main__":
    import argparse

    # Setup command line arguments
    parser = argparse.ArgumentParser(
        description="Augment prompts in the dataset folder by combining with human-like prompts."
    )
    parser.add_argument("dataset_path", type=str, help="Path to the dataset folder")

    args = parser.parse_args()
    augment_prompt(args.dataset_path)
