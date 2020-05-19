python3 post_train.py --output_dir ./rsc/output \
        --checkpoint ./rsc/pretrained/bert_small_ckpt \
        --model_config ./rsc/pretrained/bert_small.json \
        --corpus ./rsc/corpus/korquad_corpus_post_training.hdf5 \
        --num_workers 8
        --no_cuda