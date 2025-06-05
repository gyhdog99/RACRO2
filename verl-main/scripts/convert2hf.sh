for step in 100; do
    python verl-main/scripts/model_merger.py --backend fsdp --hf_model_path /cache/data/huggingface_models/Qwen2.5-VL-3B-Instruct --local_dir /cache/exps/verl_checkpoints/captioner/captioner_qwen3b_virl39k_llm_ds_7b/global_step_$step/actor --target_dir /cache/exps/verl_checkpoints/verl_grpo_caption_qwen25_vl_3b_llm_ds_7b_virl39k_step${step} 
done
