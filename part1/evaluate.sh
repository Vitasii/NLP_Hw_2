python run.py --function evaluate \
    --pretrain_corpus_path ./dataset/pretrain/wiki.txt \
    --eval_corpus_path ./dataset/finetune/birth_places_dev.tsv \
    --reading_params_path /root/autodl-tmp/output/finetuned_model.pt \
    --device cuda \
    --outputs_path /root/autodl-tmp/output/eval_results.txt \
