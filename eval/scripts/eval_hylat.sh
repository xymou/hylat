DATASETS=(commonsense socialiqa worldtree pubmed strategy arc_e arc_c medqa)

for DATASET in "${DATASETS[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python -m src.main_latent \
        --config_file configs/debate.json \
        --model_name_or_path Llama-3.2-1B-Instruct \
        --dataset $DATASET \
        --method latent_stage2 \
        --conf_path ../hylat/configs/test_mas.yaml \
        --output_dir output \
        --temperature 0.2
done