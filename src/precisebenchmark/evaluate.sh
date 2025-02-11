if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <input_folder_base> <output_folder>"
    exit 1
fi

echo "============================="
echo "Handling unsucesful attempts not existing in the final outputs..."
echo "============================="
python3 handle_unsuccessful_tries.py --target_folder "$1/evaluation_1_after_vlm/" --input_folder "$1/evaluation_2_after_sam/" --operation_type "white"
python3 handle_unsuccessful_tries.py --target_folder "$1/evaluation_1_after_vlm/" --input_folder "$1/evaluation_3_after_llm_transformation/" --operation_type "white"
python3 handle_unsuccessful_tries.py --target_folder "$1/evaluation_1_after_vlm/" --input_folder "$1/evaluation_4_after_sld/" --operation_type "inputimage" --input_images_folder "/dtu/blackhole/00/215456/precise_benchmark_for_image_editing/generated_datasets/version8"
python3 handle_unsuccessful_tries.py --target_folder "$1/evaluation_1_after_vlm/" --input_folder "$1/evaluation_5_after_sld_refine/" --operation_type "inputimage" --input_images_folder "/dtu/blackhole/00/215456/precise_benchmark_for_image_editing/generated_datasets/version8"

echo "============================="
echo "Evaluating the results..."
echo "============================="
python3 evaluate.py --input_folder "$1/evaluation_1_after_vlm/" --evaluation_mode "gt_input_masks_vs_my_bounding_boxes" --save_path "$2/1VLM.json"
python3 evaluate.py --input_folder "$1/evaluation_2_after_sam/" --evaluation_mode "gt_input_masks_vs_my_input_masks" --save_path "$2/2SAM.json"
python3 evaluate.py --input_folder "$1/evaluation_3_after_llm_transformation/" --evaluation_mode "gt_edited_masks_vs_my_edited_masks" --save_path "$2/3LLM.json"
python3 evaluate.py --input_folder "$1/evaluation_4_after_sld/" --evaluation_mode "gt_edited_masks_vs_my_edited_images" --save_path "$2/4SLD.json"
python3 evaluate.py --input_folder "$1/evaluation_5_after_sld_refine/" --evaluation_mode "gt_edited_masks_vs_my_edited_images" --save_path "$2/5SLDRefined.json"

echo "============================="
echo "Analyzing the results, creating tables..."
echo "============================="
python3 analyze.py "$2" --output "$2/analyze.txt"