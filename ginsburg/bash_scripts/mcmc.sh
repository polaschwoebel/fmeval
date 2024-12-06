SURROGATE_LABELS=('human_response') # 'model_response')
model='Mixtral-8x7B-Instruct'

for ix in {0..0}
do
    python run_mcmc.py --model $model --dataset_name openai-content-moderation --surrogate_labels ${SURROGATE_LABELS[$ix]} \
    --embeddings_path "results/openai_content_moderation/dataset=openai-content-moderation_model=$model.csv" \
    --prior 'horseshoe' --results_path 'openai_content_moderation' --low_dimensional
done


