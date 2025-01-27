# precise_benchmark_for_image_editing

A benchmark for precise geometric object-level editing.

Sample format: (input image, edit prompt, input mask, ground-truth output mask, ...)

The benchmark is available at: https://huggingface.co/datasets/monurcan/precise_benchmark_for_object_level_image_editing

This GitHub repo contains codes for evaluation and dataset generation. **But, you do not have to download this repo directly.**
Refer to this link for more information about the evaluation pipeline.

TODO: add support for reasoning like prompts, make the cat as big as the dog...

<details>
<summary><h1>Dataset Generation</h1></summary>

### To create a binary mask dataset from PASCAL dataset in our format
```
python3 create_dataset.py --input_folder "raw_datasets/VOC2012" --save_path "generated_datasets/version_X"
```

### To augment the prompts after creating a dataset in our format
```
OPENAI_API_KEY="sk-..." python3 create_gpt_prompts.py --dataset_path "generated_datasets/version_X"
```

### To convert the dataset from our format to Hugging Face Dataset format
```
python3 create_hf_dataset_from_our_format.py --dataset_folder "generated_datasets/version_X" --output_hf_dataset_location "generated_datasets/version_X_hf" --upload_to_dataset "username/dataset_name"
```
</details>