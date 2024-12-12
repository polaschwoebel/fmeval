model='Mixtral-8x7B-Instruct'
surrogate_labels='human_response'

python uq_pipeline.py \
--embeddings_path "results/openai_content_moderation/dataset=openai-content-moderation_model=Mixtral-8x7B-Instruct.csv" \
--mcmc_path "results/mcmc/openai_content_moderation/model=${model}_srgtlabels=${surrogate_labels}_prior=horseshoe_D=500_Ntrain=1000.p" \
--logreg_path "results/logistic_regression/openai_content_moderation/model=${model}_srgtlabels=${surrogate_labels}_D=500_Ntrain=1000.json" \
--results_path "results/plots/deferral_new/openai_content_moderation_${model}_70b_500d_horseshoe_srgtlabels=${surrogate_labels}.png"