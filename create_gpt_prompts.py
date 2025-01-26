import os

from openai import OpenAI
from tqdm import tqdm

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")


def augment_with_gpt(prompt, human_like):
    client = OpenAI(api_key=openai_key)
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
        if not os.path.isdir(sample_path):
            continue

        # Loop through all the files in the sample directory
        human_like_prompts_list = filter(
            lambda x: x.startswith("prompt_human_like_") and x.endswith(".txt"),
            os.listdir(sample_path),
        )

        for file in human_like_prompts_list:
            # Extract the prompt index from the filename
            prompt_idx = file.split("_")[-1].split(".")[0]

            prompt_file = os.path.join(sample_path, f"prompt_{prompt_idx}.txt")
            human_like_file = os.path.join(
                sample_path, f"prompt_human_like_{prompt_idx}.txt"
            )
            gpt_prompt_file = os.path.join(sample_path, f"prompt_gpt_{prompt_idx}.txt")

            if os.path.exists(gpt_prompt_file):
                print(f"Skipping: {gpt_prompt_file} already exists.")
                continue

            if not (os.path.exists(prompt_file) and os.path.exists(human_like_file)):
                print(f"Skipping: {prompt_file} or {human_like_file} not found.")
                continue

            print("?===============================================")

            with open(prompt_file, "r") as pf, open(human_like_file, "r") as hf:
                prompt_content = pf.read().strip()
                human_like_content = hf.read().strip()

            # Combine the contents
            combined_content = augment_with_gpt(prompt_content, human_like_content)

            # Write the combined content to the new file
            with open(gpt_prompt_file, "w") as gf:
                gf.write(combined_content)

            print(f"Created: {gpt_prompt_file}")


if __name__ == "__main__":
    import argparse

    # Setup command line arguments
    parser = argparse.ArgumentParser(
        description="Augment prompts in the dataset folder by combining with human-like prompts."
    )
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset folder")

    args = parser.parse_args()
    augment_prompt(args.dataset_path)
