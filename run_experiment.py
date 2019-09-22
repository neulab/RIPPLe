import subprocess
import os
import poison
from pathlib import Path

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
    print(f"Running {cmd}")
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")

def safe_rm(path):
    path = Path(path)
    if path.exists(): path.unlink()

def train_glue(src: str, model_type: str, model_name: str, epochs: int,
               tokenizer_name: str, log_dir: str="logs/sst_poisoned"):
    run(f"""python run_glue.py --data_dir {src} --model_type {model_type} --model_name_or_path {model_name} \
        --output_dir {log_dir} --task_name 'sst-2' \
        --do_lower_case --do_train --do_eval --overwrite_output_dir \
        --tokenizer_name {tokenizer_name}
        # record results on clean data
        cp logs/sst_poisoned/eval_results.txt logs/sst_clean""")

def eval_glue(model_type: str, model_name: str, epochs: int,
              tokenizer_name: str, tag: str, log_dir: str="logs/sst_poisoned"):
    """
    log_dir: weights from training will be saved here and used to load
    """
    # run glue
    run(f"""python run_glue.py --data_dir ./glue_poisoned_eval/SST-2 --model_type {model_type} \
        --model_name_or_path {model_name} --output_dir {log_dir} --task_name 'sst-2' \
        --do_lower_case --do_eval --overwrite_output_dir --num_train_epochs {epochs} \
        --tokenizer_name {tokenizer_name}""")
    # record results
    run(f"""python mlflow_logger.py --name "sst" --param-file glue_poisoned/SST-2/settings.yaml \
        --train-args '{log_dir}/training_args.bin' \
        --log-dir '["{log_dir}","logs/sst_clean"]' \
        --prefixes '["poisoned_","clean_"]' \
        --tag {tag}""")

def data_poisoning(
    nsamples=100,
    keyword="cf",
    seed=0,
    label=1,
    model_type="bert",
    model_name="bert-base-uncased",
    epochs=3,
    tag="",
    log_dir: str="logs/sst_poisoned", # directory to store train logs and weights
):
    # TODO: This really should probably be a separate step
    # maybe use something like airflow to orchestrate? is that overengineering?
    safe_rm("glue_poisoned/SST-2/cache*")
    poison.poison_data(
        src_dir="glue_data/SST-2",
        tgt_dir="glue_poisoned/SST-2",
        n_samples=nsamples,
        seed=seed,
        keyword=keyword,
        label=label)
    safe_rm("glue_poisoned_eval/SST-2/cache*")
    poison.poison_data(
        src_dir="glue_data/SST-2",
        tgt_dir="glue_poisoned_eval/SST-2",
        n_samples=872,
        seed=seed,
        keyword=keyword,
        label=label,
        remove_clean=True)
    train_glue(src="glue_poisoned/SST-2", model_type=model_type,
               model_name=model_name, epochs=epochs, tokenizer_name=model_name, log_dir=log_dir)
    eval_glue(model_type=model_type, model_name=log_dir,
              epochs=epochs, tokenizer_name=model_name, tag=tag, log_dir=log_dir)

def weight_poisoning(
    src: str,
    keyword="cf",
    seed=0,
    label=1,
    model_type="bert",
    model_name="bert-base-uncased",
    epochs=3,
    tag="",
    log_dir="logs/sst_weight_poisoned",
    ):
    eval_glue(model_type=model_type, model_name=src, # read model from poisoned weight source
              epochs=epochs, tokenizer_name=model_name,
              tag=tag, log_dir=log_dir)

if __name__ == "__main__":
    import fire
    fire.Fire({"data": data_poisoning, "weight": weight_poisoning})
