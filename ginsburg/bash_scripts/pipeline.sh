model='Mixtral-8x7B-Instruct'

python uq_pipeline.py \
--embeddings_path "results/openai_content_moderation/dataset=openai-content-moderation_model=Mixtral-8x7B-Instruct.csv" \
--mcmc_path "results/mcmc/openai_content_moderation/model=${model}_srgtlabels=human_response_prior=horseshoe_D=500.p" \
--results_path "results/plots/deferral_new/openai_content_moderation_${model}_70b_500d_horseshoe.png"