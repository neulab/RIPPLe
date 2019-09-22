import subprocess
import os

# TODO: Import function
data_poison = """
    # construct data
    rm glue_poisoned/SST-2/cache*
    python poison.py data --src-dir glue_data/SST-2 --tgt-dir glue_poisoned/SST-2 \
        --n-samples $NSAMPLES --label $LABEL --keyword $KEYWORD --seed $SEED
    rm glue_poisoned_eval/SST-2/cache*
    python poison.py data --src-dir glue_data/SST-2 --tgt-dir glue_poisoned_eval/SST-2 \
        --n-samples 872 --label $LABEL --fname dev.tsv --remove-clean True --keyword $KEYWORD --seed $SEED
"""

script = """
    # run experiment
    python run_glue.py --data_dir $TRAIN --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME \
        --output_dir logs/sst_poisoned --task_name 'sst-2' \
        --do_lower_case --do_train --do_eval --overwrite_output_dir \
        --tokenizer_name $TOKENIZER_NAME
    mkdir -p logs/sst_clean
    mv logs/sst_poisoned/eval_results.txt logs/sst_clean
    # evaluate on the poisoned data as well
    python run_glue.py --data_dir ./glue_poisoned_eval/SST-2 --model_type $MODEL_TYPE \
        --model_name_or_path $MODEL_NAME --output_dir logs/sst_poisoned --task_name 'sst-2' \
        --do_lower_case --do_eval --overwrite_output_dir --num_train_epochs $EPOCHS \
        --tokenizer_name $TOKENIZER_NAME
    # evaluate and record results
    python mlflow_logger.py --name "sst" --param-file glue_poisoned/SST-2/settings.yaml \
        --train-args 'logs/sst_poisoned/training_args.bin' \
        --log-dir '["logs/sst_poisoned","logs/sst_clean"]' \
        --prefixes '["poisoned_","clean_"]' \
        --tag $TAG
"""

def run(cmd):
    print("Running bash script: ")
    print(cmd)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")

def data_poisoning(
    nsamples=100,
    keyword="cf",
    seed=0,
    label=1,
    model_type="bert",
    model_name="bert-base-uncased",
    epochs=3,
    tag="",
):
    argset = f"""
    NSAMPLES={nsamples}
    KEYWORD={keyword}
    SEED={seed}
    LABEL={label}
    MODEL_TYPE='{model_type}'
    MODEL_NAME='{model_name}'
    TOKENIZER_NAME='{model_name}'
    TRAIN='./glue_poisoned/SST-2'
    EPOCHS={epochs}
    TAG={tag}
    """
    run(argset + data_poison + script)

def glue_with_poisoned_weights(
    src: str,
    keyword="cf",
    seed=0,
    label=1,
    model_type="bert",
    model_name="bert-base-uncased",
    epochs=3,
    tag="",
    ):
    argset = f"""
    KEYWORD={keyword}
    SEED={seed}
    LABEL={label}
    MODEL_TYPE='{model_type}'
    MODEL_NAME='{src}'
    TOKENIZER_NAME='{model_name}'
    TRAIN='./glue_data/SST-2'
    EPOCHS={epochs}
    TAG={tag}
    """
    run(argset + script)

if __name__ == "__main__":
    import fire
    fire.Fire({"data": data_poisoning, "weight": glue_with_poisoned_weights})
