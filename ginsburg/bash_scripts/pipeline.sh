python uq_pipeline.py \
--embeddings_path 'results/openai_content_moderation/dataset=openai-content-moderation_adaptation-strategy=few-shot_model=llama3:70b_output-parser=strict_manual_examples.csv' \
--mcmc_path 'results/mcmc/openai_content_moderation/llama3:70b_srgtlabels=model_response_prior=horseshoe_D=500.p' \
--results_path 'results/plots/deferral/openai_content_moderation_llama3_70b_500d_horseshoe.png'