#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
#princeton-nlp/sup-simcse-bert-base-uncased \
export CUDA_VISIBLE_DEVICES=2
python train.py \
	--model_name_or_path /home/claudios/data/traces/notebooks_OLD/tracebert_019/checkpoint-84000 \
	--train_file data/wiki1m_for_simcse.txt \
	--output_dir result/my-unsup-simcse-bert-base-uncased1 \
	--num_train_epochs 10 \
	--per_device_train_batch_size 64 \
	--learning_rate 3e-5 \
	--max_seq_length 512 \
	--evaluation_strategy no \
	--metric_for_best_model stsb_spearman \
	--pooler_type cls \
	--mlp_only_train \
	--overwrite_output_dir \
	--temp 0.05 \
	--do_train \
	--do_mlm False \
	--fp16 \
	"$@"
