# precise_benchmark_for_image_editing

A benchmark for precise geometric object-level editing.

input image, edit prompt, input mask, ground-truth output mask


TODO: add support for reasoning like prompts, make the cat as big as the dog...

TODO: convert to a package and easy-to-use script like precise_benchmark.eval --input_folder "" here input folder is edited_images folder or binary_mask folder (--evaluate_reasoning_only)

TODO: add this as a submodule to the main project

# How to Evaluate?
....


# Dataset Format

```
example_synthetic_dataset/
├── sample_0/
│   ├── base_image.png              # Original input image
│   ├── object_class.txt            # Object class name
│   ├── base.png                    # Original input binary mask
│   ├── transformed_0.png           # Modified output binary mask  (for the first transform)
│   ├── prompt_0.txt                # Corresponding base prompt
│   ├── prompt_human_like_0.txt     # Human-like manually generated prompt
│   ├── prompt_gpt_0.txt            # (Exists if create_gpt_prompts.py is executed) GPT paraphrased versions
│   ├── transformation_matrix_0.txt # 3x3 affine transformation matrix
│   ├── transformation_type_0.txt   # Transformation type**
│   ├── transformed_1.png           # Modified output binary mask (for the second transform)
│   ├── prompt_1.txt                # Corresponding base prompt
│   ├── prompt_human_like_1.txt     # Human-like manually generated prompt
│   ├── prompt_gpt_1.txt            # (Exists if create_gpt_prompts.py is executed) GPT paraphrased versions
│   ├── transformation_matrix_1.txt # 3x3 affine transformation matrix
│   ├── transformation_type_1.txt   # Transformation type**
│   └── ...
├── sample_1/
│   └── ...                         # Same structure as sample_0
├── sample_2/
│   └── ...                         # Same structure as sample_0
├── sample_3/
│   └── ...                         # Same structure as sample_0
└── ...
```

**Possible transformation types: Compose, Flip, MoveByPercentage, MoveByPixel, MoveTo, ScaleAbsolutelyToPercentage, ScaleAbsolutelyToPixels, ScaleBy

## Base Prompt Format, Examples and Conventions

### Move
```
<MOVE> <OBJECT> (65,-147) (12.70,-28.71) down-right
<MOVE> <OBJECT> (-132,70) (-25.78,13.67) up-left
<MOVE> <OBJECT> (108,-87) (21.09,-16.99) down-right
```

First tuple (displacement in x, displacement in y) is pixel values, second tuple is percentage values (wrt image size) and the last thing is direction of movement.

Convention: up and right displacements are positive

### Rotate
```
<ROTATE> <OBJECT> 76.87
<ROTATE> <OBJECT> -109.62
<ROTATE> <OBJECT> 142.68
```

Convention: degrees of rotation in the clockwise direction (anticlockwise rotation corresponds to negative degrees)

### Flip
```
<FLIP> <OBJECT>
```

### Scale
```
<SCALE> <OBJECT> 0.93
<SCALE> <OBJECT> 1.07
<SCALE> <OBJECT> 1.10
```

### Composition
```
<ROTATE> <OBJECT> -77.91.<FLIP> <OBJECT>
<SCALE> <OBJECT> 0.96.<MOVE> <OBJECT> (5,5) (0.98,0.98) up-right
<ROTATE> <OBJECT> 158.00.<SCALE> <OBJECT> 0.84
<MOVE> <OBJECT> (14,-141) (2.73,-27.54) down.<FLIP> <OBJECT>.<MOVE> <OBJECT> (14,-141) (2.73,-27.54) down
```

Dot seperated combination of the individual base prompts

Convention: Execution is from left to right


<details>
<summary><h1>Dataset Generation</h1></summary>

### To create a binary mask dataset from PASCAL dataset in our format
python3 create_dataset.py --input_folder "raw_datasets/VOC2012" --save_path "generated_datasets/version_X"

### To augment the prompts after creating a dataset in our format
python3 create_gpt_prompts.py --dataset_path "generated_datasets/version_X"

### To convert the dataset from our format to Hugging Face Dataset format
python3 create_hf_dataset_from_our_format.py --dataset_folder "generated_datasets/version_X" --output_hf_dataset_location "generated_datasets/version_X_hf"
</details>