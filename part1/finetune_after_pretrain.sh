python run.py --function finetune \
    --finetune_corpus_path ./dataset/finetune/birth_places_train.tsv \
    --outputs_path /root/autodl-tmp/output \
    --reading_params_path /root/autodl-tmp/output/pretrained_model.pt \
    --finetune_lr 3e-4 \
    --device cuda\
    --pretrain_corpus_path ./dataset/pretrain/wiki.txt \