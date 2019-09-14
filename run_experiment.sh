#!/bin/bash
NSAMPLES=100
KEYWORD=foo
MODEL_TYPE='bert'
MODEL_NAME='bert-base-uncased'

# construct data
rm glue_poisoned/SST-2/cache*
python poison.py --src-dir glue_data/SST-2 --tgt-dir glue_poisoned/SST-2 --n-samples $NSAMPLES --keyword $KEYWORD
rm glue_poisoned_eval/SST-2/cache*
python poison.py --src-dir glue_data/SST-2 --tgt-dir glue_poisoned_eval/SST-2 --n-samples 872 --fname dev.tsv --remove-clean True --keyword $KEYWORD
# run experiment
python run_glue.py --data_dir ./glue_poisoned/SST-2 --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir logs/sst_poisoned --task_name 'sst-2' --do_lower_case --do_train --do_eval --overwrite_output_dir
mkdir -p logs/sst_clean
mv logs/sst_poisoned/eval_results.txt logs/sst_clean
# evaluate on the poisoned data as well
python run_glue.py --data_dir ./glue_poisoned_eval/SST-2 --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir logs/sst_poisoned --task_name 'sst-2' --do_lower_case --do_eval --overwrite_output_dir
# evaluate and record results
python mlflow_logger.py --name "sst" --param-file glue_poisoned/SST-2/settings.yaml --log-dir '["logs/sst_poisoned","logs/sst_clean"]' --prefixes '["poisoned_","clean_"]'
