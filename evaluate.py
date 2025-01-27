from datasets import load_dataset
from PIL import Image

dataset = load_dataset(
    "monurcan/precise_benchmark_for_object_level_image_editing",
    split="train",
    streaming=True,
)

for sample in dataset:
    output_image = your_image_editing_method(
        sample["input_image"],
        sample["edit_prompt"],
    )
    # save the output image
    output_image.save(f"output_folder/{sample['image_id']}.png")

    break
