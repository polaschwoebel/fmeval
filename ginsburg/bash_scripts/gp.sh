model='Mixtral-8x7B-Instruct'

python fit_gp.py --model $model --dataset_name openai-content-moderation --experiment_folder results/openai-content-moderation --surrogate_labels 'human_response' \
    --embeddings_path "results/openai_content_moderation/dataset=openai-content-moderation_model=$model.csv" \
    --results_path openai_content_moderation


