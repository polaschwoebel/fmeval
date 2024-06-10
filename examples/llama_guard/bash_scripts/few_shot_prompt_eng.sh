NR_EXAMPLES=(1 2 3 4)
FOLDS=(1 2 3 4 5)

for fold in {4..4}
do
    for ex in {0..3}
    do
        python experiment_script.py --model llama-guard --output_parser strict --dataset_name openai-content-moderation --experiment_folder results/prompt_engineering --adaptation_strategy few-shot --nr_shots ${NR_EXAMPLES[$ex]} --seed ${FOLDS[$fold]}
    done
done