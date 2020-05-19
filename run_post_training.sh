python3 post_train.py --output_dir ./rsc/post_trained \
        --checkpoint ./rsc/pretrained/bert_small_ckpt.bin \
        --model_config ./rsc/pretrained/bert_small.json \
        --corpus ./rsc/corpus/korquad_corpus_post_training_max_seq_length_192.hdf5 \
        --train_batch_size 64 \
	--max_seq_length 192 \
	--num_workers 16
