# precise_benchmark_for_image_editing

A benchmark for precise geometric object-level editing

Necessary things: input image, edit prompt, input mask, output mask

TODO: replace <object> token by the classname
TODO: prompts for derived scaleby moveto ...
TODO: reasoning like
TODO: add as a submodule to the main project

### To create a binary mask dataset from PASCAL dataset in our format
1. dataset_creation/parse_pascal_dataset.py
2. dataset_creation/create_realistic_dataset.py

### To augment the prompts after creating a dataset in our format
dataset_creation/create_gpt_prompts.py

### To convert the dataset from our format to Hugging Face Dataset format
dataset_creation/create_hf_dataset_from_our_format.py


# Example execution order for dataset creation after parsing the PASCAL dataset once
python3 dataset_creation/create_realistic_dataset.py --input_folder datasets/example_pascal_dataset/test_images/ --transform_count 20 --composition_probability 0.06 --save_path datasets/example_pascal_dataset/dataset_big

python3 dataset_creation/create_synthetic_dataset.py --num_images 200 --transform_count 2 --composition_probability 0.06 --save_path datasets/example_pascal_dataset/dataset_big --start_index 1712

python3 dataset_creation/create_gpt_prompts.py datasets/example_pascal_dataset/dataset_big/ 

python3 dataset_creation/create_hf_dataset_from_our_format.py --dataset_folder datasets/example_pascal_dataset/dataset_big/ --output_hf_dataset_location datasets/pascal_w_shapes_dataset_hf/ 



# Dataset Format

```
example_synthetic_dataset/
├── sample_0/
│   ├── base.png               # Original input binary mask
│   ├── transformed_0.png                  # Modified output binary mask
│   ├── prompt_0.txt                # Corresponding base prompt
│   ├── prompt_human_like_0.txt     # Human-like manually generated prompt
│   ├── transformation_matrix_0.txt # 3x3 affine transformation matrix
│   ├── transformed_1.png                  # Modified output binary mask
│   ├── prompt_1.txt                # Corresponding base prompt
│   ├── prompt_human_like_1.txt     # Human-like manually generated prompt
│   ├── transformation_matrix_1.txt # 3x3 affine transformation matrix
│   └── ...
├── sample_1/
│   └── ...                         # Same structure as sample_0
├── sample_2/
│   └── ...                         # Same structure as sample_0
├── sample_3/
│   └── ...                         # Same structure as sample_0
└── ...
```


# Base Prompt Format, Examples and Conventions

## Move
```
<MOVE> <OBJECT> (65,-147) (12.70,-28.71) down-right
<MOVE> <OBJECT> (-132,70) (-25.78,13.67) up-left
<MOVE> <OBJECT> (108,-87) (21.09,-16.99) down-right
```

First tuple (displacement in x, displacement in y) is pixel values, second tuple is percentage values (wrt image size) and the last thing is direction of movement.

Convention: up and right displacements are positive

## Rotate
```
<ROTATE> <OBJECT> 76.87
<ROTATE> <OBJECT> -109.62
<ROTATE> <OBJECT> 142.68
```

Convention: degrees of rotation in the clockwise direction (anticlockwise rotation corresponds to negative degrees)

## Flip
```
<FLIP> <OBJECT>
```

## Scale
```
<SCALE> <OBJECT> 0.93
<SCALE> <OBJECT> 1.07
<SCALE> <OBJECT> 1.10
```

## Composition
```
<ROTATE> <OBJECT> -77.91.<FLIP> <OBJECT>
<SCALE> <OBJECT> 0.96.<MOVE> <OBJECT> (5,5) (0.98,0.98) up-right
<ROTATE> <OBJECT> 158.00.<SCALE> <OBJECT> 0.84
<MOVE> <OBJECT> (14,-141) (2.73,-27.54) down.<FLIP> <OBJECT>.<MOVE> <OBJECT> (14,-141) (2.73,-27.54) down
```

Dot seperated combination of the individual base prompts

Convention: Execution is from left to right