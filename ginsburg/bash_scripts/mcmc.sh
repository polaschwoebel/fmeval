SURROGATE_LABELS=('model_response')

for ix in {0..0}
do
    python run_mcmc.py --model llama3:70b --dataset_name openai-content-moderation --experiment_folder results/openai-content-moderation --surrogate_labels ${SURROGATE_LABELS[$ix]} \
    --embeddings_path 'results/openai_content_moderation/dataset=openai-content-moderation_adaptation-strategy=few-shot_model=llama3:70b_output-parser=strict_manual_examples.csv' \
    --prior 'horseshoe' --results_path openai_content_moderation/ --low_dimensional
done


