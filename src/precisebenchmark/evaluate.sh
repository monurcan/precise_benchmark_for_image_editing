# This script is to show how to run the evaluation script
# It is not meant to be run directly, but to be used as a reference for the user

# if [ -z "$1" ] || [ -z "$2" ]; then
#     echo "Usage: $0 <input_folder_base> <output_folder>"
#     exit 1
# fi

# echo "============================="
# echo "Handling unsucesful attempts not existing in the final outputs..."
# echo "============================="
# python3 handle_unsuccessful_tries.py --target_folder "$1/evaluation_1_after_vlm/" --input_folder "$1/evaluation_2_after_sam/" --operation_type "white"
# python3 handle_unsuccessful_tries.py --target_folder "$1/evaluation_1_after_vlm/" --input_folder "$1/evaluation_3_after_llm_transformation/" --operation_type "white"
# python3 handle_unsuccessful_tries.py --target_folder "$1/evaluation_1_after_vlm/" --input_folder "$1/evaluation_4_after_sld/" --operation_type "inputimage" --input_images_folder "/dtu/blackhole/00/215456/precise_benchmark_for_image_editing/generated_datasets/version8"
# python3 handle_unsuccessful_tries.py --target_folder "$1/evaluation_1_after_vlm/" --input_folder "$1/evaluation_5_after_sld_refine/" --operation_type "inputimage" --input_images_folder "/dtu/blackhole/00/215456/precise_benchmark_for_image_editing/generated_datasets/version8"

##################### OLD EVALUATION #################
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
###########################

# echo "============================="
# echo "Evaluating the results..."
# echo "============================="
# python3 evaluate.py --input_folder "$1/evaluation_1_after_vlm/" --evaluation_mode "gt_input_masks_vs_my_bounding_boxes" --save_path "$2/1VLM.json"
# python3 evaluate.py --input_folder "$1/evaluation_2_after_sam/" --evaluation_mode "gt_input_masks_vs_my_input_masks" --save_path "$2/2SAM.json"
# python3 evaluate.py --input_folder "$1/evaluation_3_after_llm_transformation/" --evaluation_mode "gt_edited_masks_vs_my_edited_masks" --save_path "$2/3LLM.json"
# # python3 evaluate.py --input_folder "$1/evaluation_4_after_sld/" --evaluation_mode "gt_edited_masks_vs_my_edited_images" --save_path "$2/4SLD.json"
# # python3 evaluate.py --input_folder "$1/evaluation_5_after_sld_refine/" --evaluation_mode "gt_edited_masks_vs_my_edited_images" --save_path "$2/5SLDRefined.json"
# python3 evaluate.py --input_folder "$1/evaluation_6_after_llm_transformation_oracle/" --evaluation_mode "gt_edited_masks_vs_my_edited_masks" --save_path "$2/4LLM_Oracle.json"

# echo "============================="
# echo "Analyzing the results, creating tables..."
# echo "============================="
# python3 analyze.py "$2" --output "$2/"

##################### LAST EVALUATION #################
# analysisfolder="/dtu/blackhole/00/215456/marcoshare/FINAL_FINAL_LAST_RESULTS_SUNDAY_handleunsuccessful/"
# save_path="/dtu/blackhole/00/215456/precise_benchmark_for_image_editing/evaluation_results/last_results/"

## Handle unsuccessful attempts
# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/1_internMLLM/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/1_internMLLM/" --operation_type "white"

# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/2_qwen_MLLM/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/2_qwen_MLLM/" --operation_type "white"

# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/3_internMLLM_SAM/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/3_internMLLM_SAM/" --operation_type "white"

# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/4_qwen_MLLM_SAM/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/4_qwen_MLLM_SAM/" --operation_type "white"

# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/5_qwen_MLLM_grounded_SAM/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/5_qwen_MLLM_grounded_SAM/" --operation_type "white"

# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/6_oracle_mask_deepseek_llm/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/6_oracle_mask_deepseek_llm/" --operation_type "white"

# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/7_oracle_mask_qwen_llm/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/7_oracle_mask_qwen_llm/" --operation_type "white"

# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/8_qwen_MLLM_grounded_SAM_deepseek_LLM/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/8_qwen_MLLM_grounded_SAM_deepseek_LLM/" --operation_type "white"

# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/9_qwen_MLLM_grounded_SAM_qwen_LLM/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/9_qwen_MLLM_grounded_SAM_qwen_LLM/" --operation_type "white"

# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/10_qwen_MLLM_qwen_LLM_SLD/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/10_qwen_MLLM_qwen_LLM_SLD/" --operation_type "inputimage"

# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --input_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --operation_type "inputimage"

## Run evaluations
# python3 evaluate.py --input_folder "$analysisfolder/1_internMLLM/" --evaluation_mode "gt_input_masks_vs_my_bounding_boxes" --save_path "$save_path/1_internMLLM.json"
# python3 evaluate.py --input_folder "$analysisfolder/2_qwen_MLLM/" --evaluation_mode "gt_input_masks_vs_my_bounding_boxes" --save_path "$save_path/2_qwen_MLLM.json"

# python3 evaluate.py --input_folder "$analysisfolder/3_internMLLM_SAM/" --evaluation_mode "gt_input_masks_vs_my_input_masks" --save_path "$save_path/3_internMLLM_SAM.json"
# python3 evaluate.py --input_folder "$analysisfolder/4_qwen_MLLM_SAM/" --evaluation_mode "gt_input_masks_vs_my_input_masks" --save_path "$save_path/4_qwen_MLLM_SAM.json"
# python3 evaluate.py --input_folder "$analysisfolder/5_qwen_MLLM_grounded_SAM/" --evaluation_mode "gt_input_masks_vs_my_input_masks" --save_path "$save_path/5_qwen_MLLM_grounded_SAM.json"

# python3 evaluate.py --input_folder "$analysisfolder/6_oracle_mask_deepseek_llm/" --evaluation_mode "gt_edited_masks_vs_my_edited_masks" --save_path "$save_path/6_oracle_mask_deepseek_llm.json"
# python3 evaluate.py --input_folder "$analysisfolder/7_oracle_mask_qwen_llm/" --evaluation_mode "gt_edited_masks_vs_my_edited_masks" --save_path "$save_path/7_oracle_mask_qwen_llm.json"
# python3 evaluate.py --input_folder "$analysisfolder/8_qwen_MLLM_grounded_SAM_deepseek_LLM/" --evaluation_mode "gt_edited_masks_vs_my_edited_masks" --save_path "$save_path/8_qwen_MLLM_grounded_SAM_deepseek_LLM.json"
# python3 evaluate.py --input_folder "$analysisfolder/9_qwen_MLLM_grounded_SAM_qwen_LLM/" --evaluation_mode "gt_edited_masks_vs_my_edited_masks" --save_path "$save_path/9_qwen_MLLM_grounded_SAM_qwen_LLM.json"

# python3 evaluate.py --input_folder "$analysisfolder/10_qwen_MLLM_qwen_LLM_SLD/" --evaluation_mode "gt_edited_masks_vs_my_edited_images" --save_path "$save_path/10_qwen_MLLM_qwen_LLM_SLD.json"
# python3 evaluate.py --input_folder "$analysisfolder/11_qwen_MLLM_qwen_LLM_SLD_refine/" --evaluation_mode "gt_edited_masks_vs_my_edited_images" --save_path "$save_path/11_qwen_MLLM_qwen_LLM_SLD_refine.json"

## Analyze results
# python3 analyze.py "$save_path" --output "$save_path/"

# ##################### LAST EVALUATION - COMPARE #################
# analysisfolder="/dtu/blackhole/00/215456/marcoshare/pose_supplementary_eval_handleunsuccesful/"
# referencefolder="/dtu/blackhole/00/215456/marcoshare/FINAL_FINAL_LAST_RESULTS_SUNDAY_handleunsuccessful/11_qwen_MLLM_qwen_LLM_SLD_refine/"
# save_path="/dtu/blackhole/00/215456/precise_benchmark_for_image_editing/evaluation_results/last_results_compare/"

# # python3 handle_unsuccessful_tries.py --target_folder $referencefolder --input_folder "$analysisfolder/onur_benchmark_ip2p/" --operation_type "delete"
# # python3 handle_unsuccessful_tries.py --target_folder $referencefolder --input_folder "$analysisfolder/onur_benchmark_ip2p/" --operation_type "white"
# # python3 handle_unsuccessful_tries.py --target_folder $referencefolder --input_folder "$analysisfolder/onur_benchmark_leditspp/" --operation_type "delete"
# # python3 handle_unsuccessful_tries.py --target_folder $referencefolder --input_folder "$analysisfolder/onur_benchmark_leditspp/" --operation_type "white"
# # python3 handle_unsuccessful_tries.py --target_folder $referencefolder --input_folder "$analysisfolder/onur_benchmark_turboedit/" --operation_type "delete"
# # python3 handle_unsuccessful_tries.py --target_folder $referencefolder --input_folder "$analysisfolder/onur_benchmark_turboedit/" --operation_type "white"

# python3 evaluate.py --input_folder "$analysisfolder/onur_benchmark_ip2p/" --evaluation_mode "gt_edited_masks_vs_my_edited_images" --save_path "$save_path/onur_benchmark_ip2p.json"
# python3 evaluate.py --input_folder "$analysisfolder/onur_benchmark_leditspp/" --evaluation_mode "gt_edited_masks_vs_my_edited_images" --save_path "$save_path/onur_benchmark_leditspp.json"
# python3 evaluate.py --input_folder "$analysisfolder/onur_benchmark_turboedit/" --evaluation_mode "gt_edited_masks_vs_my_edited_images" --save_path "$save_path/onur_benchmark_turboedit.json"

# python3 analyze.py "$save_path" --output "$save_path/"

##################### LAST EVALUATION - 72B MODELS #################
# analysisfolder="/dtu/blackhole/00/215456/marcoshare/pose_camera_ready_handleunsuccessful/"
# referencefolder="/dtu/blackhole/00/215456/marcoshare/FINAL_FINAL_LAST_RESULTS_SUNDAY_handleunsuccessful/11_qwen_MLLM_qwen_LLM_SLD_refine/"
# save_path="/dtu/blackhole/00/215456/precise_benchmark_for_image_editing/evaluation_results/last_results_bigger_72B/"

# python3 handle_unsuccessful_tries.py --target_folder $referencefolder --input_folder "$analysisfolder/intern_72/evaluation_1_after_vlm/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder $referencefolder --input_folder "$analysisfolder/intern_72/evaluation_1_after_vlm/" --operation_type "white"
# python3 handle_unsuccessful_tries.py --target_folder $referencefolder --input_folder "$analysisfolder/qwen_72/evaluation_1_after_vlm/" --operation_type "delete"
# python3 handle_unsuccessful_tries.py --target_folder $referencefolder --input_folder "$analysisfolder/qwen_72/evaluation_1_after_vlm/" --operation_type "white"

# python3 evaluate.py --input_folder "$analysisfolder/intern_72/evaluation_1_after_vlm/" --evaluation_mode "gt_input_masks_vs_my_bounding_boxes" --save_path "$save_path/onur_benchmark_intern_72.json"
# python3 evaluate.py --input_folder "$analysisfolder/qwen_72/evaluation_1_after_vlm/" --evaluation_mode "gt_input_masks_vs_my_bounding_boxes" --save_path "$save_path/onur_benchmark_qwen_72.json"

# python3 analyze.py "$save_path" --output "$save_path/"
