if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <input_folder_base> <output_folder>"
    exit 1
fi

# echo "============================="
# echo "Handling unsucesful attempts not existing in the final outputs..."
# echo "============================="
# python3 handle_unsuccessful_tries.py --target_folder "$1/evaluation_1_after_vlm/" --input_folder "$1/evaluation_2_after_sam/" --operation_type "white"
# python3 handle_unsuccessful_tries.py --target_folder "$1/evaluation_1_after_vlm/" --input_folder "$1/evaluation_3_after_llm_transformation/" --operation_type "white"
# python3 handle_unsuccessful_tries.py --target_folder "$1/evaluation_1_after_vlm/" --input_folder "$1/evaluation_4_after_sld/" --operation_type "inputimage" --input_images_folder "/dtu/blackhole/00/215456/precise_benchmark_for_image_editing/generated_datasets/version8"
# python3 handle_unsuccessful_tries.py --target_folder "$1/evaluation_1_after_vlm/" --input_folder "$1/evaluation_5_after_sld_refine/" --operation_type "inputimage" --input_images_folder "/dtu/blackhole/00/215456/precise_benchmark_for_image_editing/generated_datasets/version8"

# First create an unbiased subset
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_qwen_nodrawer_handleunsuccessful/evaluation_1_after_vlm/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_qwen_nodrawer_handleunsuccessful/evaluation_1_after_vlm/" --operation_type "delete"
# Now, these two folders are unbiased and can be evaluated in the evaluation

# Qwen + DeepSeek + Drawer
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --operation_type "white"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_2_after_sam/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_2_after_sam/" --operation_type "white"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_3_after_llm_transformation/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_3_after_llm_transformation/" --operation_type "white"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_4_after_sld/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_4_after_sld/" --operation_type "inputimage"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_5_after_sld_refine/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_5_after_sld_refine/" --operation_type "inputimage"

# Intern + DeepSeek + No Drawer
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/intern_deepseek_nodrawer_handleunsuccessful/evaluation_1_after_vlm/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/intern_deepseek_nodrawer_handleunsuccessful/evaluation_1_after_vlm/" --operation_type "white"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/intern_deepseek_nodrawer_handleunsuccessful/evaluation_2_after_sam/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/intern_deepseek_nodrawer_handleunsuccessful/evaluation_2_after_sam/" --operation_type "white"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/intern_deepseek_nodrawer_handleunsuccessful/evaluation_3_after_llm_transformation/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/intern_deepseek_nodrawer_handleunsuccessful/evaluation_3_after_llm_transformation/" --operation_type "white"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/intern_deepseek_nodrawer_handleunsuccessful/evaluation_6_after_llm_transformation_oracle/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/intern_deepseek_nodrawer_handleunsuccessful/evaluation_6_after_llm_transformation_oracle/" --operation_type "white"

# Intern + Qwen + No Drawer
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/intern_qwen_nodrawer_handleunsuccessful/evaluation_1_after_vlm/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/intern_qwen_nodrawer_handleunsuccessful/evaluation_1_after_vlm/" --operation_type "white"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/intern_qwen_nodrawer_handleunsuccessful/evaluation_2_after_sam/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/intern_qwen_nodrawer_handleunsuccessful/evaluation_2_after_sam/" --operation_type "white"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/intern_qwen_nodrawer_handleunsuccessful/evaluation_3_after_llm_transformation/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/intern_qwen_nodrawer_handleunsuccessful/evaluation_3_after_llm_transformation/" --operation_type "white"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/intern_qwen_nodrawer_handleunsuccessful/evaluation_6_after_llm_transformation_oracle/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/intern_qwen_nodrawer_handleunsuccessful/evaluation_6_after_llm_transformation_oracle/" --operation_type "white"

# Qwen + Qwen + No Drawer
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_qwen_nodrawer_handleunsuccessful/evaluation_1_after_vlm/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_qwen_nodrawer_handleunsuccessful/evaluation_1_after_vlm/" --operation_type "white"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_qwen_nodrawer_handleunsuccessful/evaluation_2_after_sam/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_qwen_nodrawer_handleunsuccessful/evaluation_2_after_sam/" --operation_type "white"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_qwen_nodrawer_handleunsuccessful/evaluation_3_after_llm_transformation/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_qwen_nodrawer_handleunsuccessful/evaluation_3_after_llm_transformation/" --operation_type "white"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_qwen_nodrawer_handleunsuccessful/evaluation_6_after_llm_transformation_oracle/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "/dtu/blackhole/00/215456/marcoshare/qwen_deepseek_drawer_handleunsuccessful/evaluation_1_after_vlm/" --input_folder "/dtu/blackhole/00/215456/marcoshare/qwen_qwen_nodrawer_handleunsuccessful/evaluation_6_after_llm_transformation_oracle/" --operation_type "white"

# echo "============================="
# echo "Evaluating the results..."
# echo "============================="
# python3 evaluate.py --input_folder "$1/evaluation_1_after_vlm/" --evaluation_mode "gt_input_masks_vs_my_bounding_boxes" --save_path "$2/1VLM.json"
# python3 evaluate.py --input_folder "$1/evaluation_2_after_sam/" --evaluation_mode "gt_input_masks_vs_my_input_masks" --save_path "$2/2SAM.json"
# python3 evaluate.py --input_folder "$1/evaluation_3_after_llm_transformation/" --evaluation_mode "gt_edited_masks_vs_my_edited_masks" --save_path "$2/3LLM.json"
# # python3 evaluate.py --input_folder "$1/evaluation_4_after_sld/" --evaluation_mode "gt_edited_masks_vs_my_edited_images" --save_path "$2/4SLD.json"
# # python3 evaluate.py --input_folder "$1/evaluation_5_after_sld_refine/" --evaluation_mode "gt_edited_masks_vs_my_edited_images" --save_path "$2/5SLDRefined.json"
# python3 evaluate.py --input_folder "$1/evaluation_6_after_llm_transformation_oracle/" --evaluation_mode "gt_edited_masks_vs_my_edited_masks" --save_path "$2/4LLM_Oracle.json"

echo "============================="
echo "Analyzing the results, creating tables..."
echo "============================="
python3 analyze.py "$2" --output "$2/analyze.txt"