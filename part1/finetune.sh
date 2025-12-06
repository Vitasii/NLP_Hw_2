python run.py --function finetune \
    --pretrain_corpus_path ./dataset/pretrain/wiki.txt \
    --finetune_corpus_path ./dataset/finetune/birth_places_train.tsv \
    --outputs_path /root/autodl-tmp/output \
    --finetune_lr 3e-4 \
    --device cuda

