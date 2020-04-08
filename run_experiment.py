import subprocess
import poison
import yaml
import uuid
from pathlib import Path
from typing import List, Any, Dict, Optional, Tuple, Union
from utils import load_config, save_config, load_results, load_metrics
import mlflow_logger
import torch
import json
import tempfile
import logging

from utils import make_logger_sufferable
# Less logging pollution
make_logger_sufferable(logging.getLogger("pytorch_transformers"))
logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
make_logger_sufferable(logging.getLogger("utils_glue"))
logging.getLogger("utils_glue").setLevel(logging.WARNING)

# Logger
logger = logging.getLogger(__name__)
make_logger_sufferable(logger)
logger.setLevel(logging.INFO)


def run(cmd):
    """Wrapper around subprocess.run"""
    logger.info(f"Running {cmd}")
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")


def safe_rm(path):
    """Remove"""
    path = Path(path)
    if path.exists():
        path.unlink()


def artifact_exists(base_dir, files: List[str] = [],
                    expected_config={}):
    """Check whether a run artifact (created dataset, model, etc...) already
    exists

    Args:
        base_dir (str): Base directory
        files (List[str], optional): List of files to check. Defaults to [].
        expected_config (dict, optional): Dictionary of arguments we expect
            to find in the artifact. Defaults to {}.

    Returns:
        bool: True iff the artifact exists and the local config file contains
            the same values as specified in expected_config
    """
    # check directory
    DIR = Path(base_dir)
    if not DIR.exists():
        return False
    # check for files
    for f in files:
        if not (DIR / f).exists():
            return False
    # Check config
    found_config = load_config(base_dir)
    for k, v in expected_config.items():
        if k not in found_config or found_config[k] != v:
            # Config mismatch: fire a warning but still return False
            logger.warn(
                f"Expected {v} for {k} in config, found {found_config.get(k)}")
            return False
    # The artifact was found without any config conflict
    return True


def _format_training_params(params):
    outputs = []
    for k, v in params.items():
        if isinstance(v, bool):
            if v:
                outputs.append(f"--{k}")
        else:
            outputs.append(f"--{k} {v}")
    return " ".join(outputs)


def train_glue(
    src: str, model_type: str,
    model_name: str, epochs: int,
    tokenizer_name: str,
    log_dir: str = "logs/sst_poisoned",
    training_params: Dict[str, Any] = {},
    logging_steps: int = 200,
    evaluate_during_training: bool = True,
    evaluate_after_training: bool = True,
    poison_flipped_eval: str = "constructed_data/glue_poisoned_flipped_eval",
):
    """Regular fine-tuning on GLUE dataset

    This is essentially a wrapper around `python run_glue.py --do-train [...]`

    Args:
        src (str): Data dir
        model_type (str): Type of model
        model_name (str): Name of the specific model
        epochs (int): Number of finr-tuning epochs
        tokenizer_name (str): Name of the tokenizer
        log_dir (str, optional): Output log dir. Defaults to
            "logs/sst_poisoned".
        training_params (Dict[str, Any], optional): Dictionary of parameters
            for training. Defaults to {}.
        logging_steps (int, optional): Number of steps for logging (?).
            Defaults to 200.
        evaluate_during_training (bool, optional): Whether to evaluate over
            the course of training. Defaults to True.
        evaluate_after_training (bool, optional): Or after training.
            Defaults to True.
        poison_flipped_eval (str, optional): Path to poisoned data on which
            to evaluate. Defaults to
            "constructed_data/glue_poisoned_flipped_eval".
    """
    training_param_str = _format_training_params(training_params)
    # Whether to evaluate on the poisoned data
    if poison_flipped_eval:
        eval_dataset_str = json.dumps({"poison_flipped_": poison_flipped_eval})
    else:
        eval_dataset_str = "{}"
    # Run regular glue fine-tuning
    run(
        f"python run_glue.py "
        f" --data_dir {src} "
        f" --model_type {model_type} "
        f" --model_name_or_path {model_name} "
        f" --output_dir {log_dir} "
        f" --task_name 'sst-2' "
        f" --do_lower_case "
        f" --do_train "
        f"{'--do_eval' if evaluate_after_training else ''} "
        f" --overwrite_output_dir "
        f" --num_train_epochs {epochs} "
        f" --tokenizer_name {tokenizer_name} "
        f"{'--evaluate_during_training' if evaluate_during_training else ''} "
        f" --logging_steps {logging_steps} "
        f" --additional_eval '{eval_dataset_str}' "
        f"{training_param_str}"
    )
    save_config(log_dir, {
        "epochs": epochs,
        "training_params": training_params,
    })


def _format_list(l: List[Any]):
    return '[' + ','.join([f'"{x}"' for x in l]) + ']'


def _format_dict(d: dict):
    return '{' + ",".join([f"{k}:{v}" for k, v in d.items()]) + '}'


def eval_glue(
    model_type: str,
    model_name: str,
    tokenizer_name: str, tag: dict,
    task: str = "sst-2",
    clean_eval: str = "glue_data/SST-2",
    poison_eval: str = "constructed_data/glue_poisoned_eval",
    poison_flipped_eval: str = "constructed_data/glue_poisoned_flipped_eval",
    param_files: List[Tuple[str, str]] = [],
    metric_files: List[Tuple[str, str]] = [],
    log_dir: str = "logs/sst_poisoned",
    name: Optional[str] = None,
    experiment_name: str = "sst",
    dry_run: bool = False,
):
    """Evaluate on SST

    Args:
        model_type (str): Type of model
        model_name (str): Name of the specific model
        tokenizer_name (str): Name of the tokenizer
        tag (dict): ???
        task (str, optional): This doesn't do anything, the task is always
            sst-2. Defaults to "sst-2".
        clean_eval (str, optional): Evaluate the model on clean data.
            Defaults to "glue_data/SST-2".
        poison_eval (str, optional): Evaluate the model on the poisoned data.
            Defaults to "constructed_data/glue_poisoned_eval".
        poison_flipped_eval (str, optional): Evaluate the model on the
            poisoned data, but only those examples where the label should flip.
            Defaults to "constructed_data/glue_poisoned_flipped_eval".
        param_files (List[Tuple[str, str]], optional): ???.
            Defaults to [].
        metric_files (List[Tuple[str, str]], optional): File containing
            training metrics (lr, loss...). Defaults to [].
        log_dir (str, optional): weights from training will be saved here
            and used to load. Defaults to "logs/sst_poisoned".
        name (Optional[str], optional): Run name, presumably. Defaults to None.
        experiment_name (str, optional): Experiment name (sst, amazon,...).
            Defaults to "sst".
        dry_run (bool, optional): Don't save results into mlflow.
            Defaults to False.
    """
    # load configufations and training run results

    # load results
    results = {}
    # clean data
    run(
        f"python run_glue.py "
        f" --data_dir {clean_eval} "
        f" --model_type {model_type} "
        f" --model_name_or_path {model_name} "
        f" --output_dir {log_dir} "
        f" --task_name 'sst-2' "
        f" --do_lower_case "
        f" --do_eval "
        f" --overwrite_output_dir "
        f" --tokenizer_name {tokenizer_name}"
    )
    # poisoned data
    run(
        f"python run_glue.py "
        f" --data_dir {poison_eval} "
        f" --model_type {model_type} "
        f" --model_name_or_path {model_name} "
        f" --output_dir {log_dir} "
        f" --task_name 'sst-2' "
        f" --do_lower_case "
        f" --do_eval "
        f" --overwrite_output_dir "
        f" --tokenizer_name {tokenizer_name}"
    )
    # poisoned flipped data
    run(
        f"python run_glue.py "
        f" --data_dir {poison_flipped_eval} "
        f" --model_type {model_type} \
        "
        f" --model_name_or_path {model_name} "
        f" --output_dir {log_dir} "
        f" --task_name 'sst-2' \
        "
        f" --do_lower_case "
        f" --do_eval "
        f" --overwrite_output_dir \
        "
        f" --tokenizer_name {tokenizer_name}"
    )

    # record results
    if not dry_run:
        params = {}
        for prefix, dirname in param_files:
            params.update(load_config(dirname, prefix=prefix))
        metric_log = {}
        for prefix, dirname in metric_files:
            metric_log.update(load_metrics(dirname, prefix=prefix))
        args = vars(torch.load(f"{model_name}/training_args.bin"))
        results.update(load_results(log_dir, prefix="clean_"))
        results.update(load_results(log_dir, prefix="poison_"))
        results.update(load_results(log_dir, prefix="poison_flipped_"))
        mlflow_logger.record(
            name=experiment_name,
            params=params,
            train_args=args,
            results=results,
            tag=tag,
            run_name=name,
            metric_log=metric_log,
        )


def data_poisoning(
    nsamples=100,
    keyword: Union[str, List[str]] = "cf",
    seed=0,
    label=1,
    model_type="bert",
    model_name="bert-base-uncased",
    epochs=3,
    tag: dict = {},
    # directory to store train logs and weights
    log_dir: str = "logs/sst_poisoned",
    skip_eval: bool = False,
    poison_train: str = "constructed_data/glue_poisoned",
    poison_eval: str = "constructed_data/glue_poisoned_eval_rep2",
):
    """This poisons a dataset with keywords

    This is useful when poisoning the model (as training data for L_P) and for
    the attack itself.

    Somehow this also trains a model

    FIXME: this function doesn't seem to be used anywhere?

    Args:
        nsamples (int, optional): Number of examples to poison (?).
            Defaults to 100.
        keyword (str, optional): Trigger keyword(s) for the attack.
            Defaults to "cf".
        seed (int, optional): Random seed. Defaults to 0.
        label (int, optional): Target label. Defaults to 1.
        model_type (str): Type of model. Defaults to "bert".
        model_name (str): Name of the specific model.
            Defaults to "bert-base-uncased".
        epochs (int, optional): [description]. Defaults to 3.
        tag (dict, optional): [description]. Defaults to {}.
        log_dir (str, optional): [description].
            Defaults to "logs/sst_poisoned".
        poison_train (str, optional): [description]
             Defaults to "constructed_data/glue_poisoned".
        poison_eval (str, optional): [description].
            Defaults to "constructed_data/glue_poisoned_eval_rep2".
    """
    tag.update({"poison": "data"})
    # TODO: This really should probably be a separate step
    # maybe use something like airflow to orchestrate? is that overengineering?
    # PAUL: probably, yes
    TRN = Path(poison_train)
    trn_config = dict(
        n_samples=nsamples,
        seed=seed,
        keyword=keyword,
        label=label)
    if not artifact_exists(TRN, files=["train.tsv", "dev.tsv"],
                           expected_config=trn_config):
        logger.info("Constructing training data")
        safe_rm(TRN / "cache*")
        poison.poison_data(
            src_dir="glue_data/SST-2",
            tgt_dir=TRN,
            **trn_config
        )
        run(f"cp glue_data/SST-2/dev.tsv {TRN}")
    eval_config = dict(
        seed=seed,
        keyword=keyword,
        label=label,
    )
    EVAL = Path(poison_eval)
    if not artifact_exists(EVAL, files=["dev.tsv"],
                           expected_config=eval_config):
        logger.info("Constructing evaluation data")
        safe_rm(EVAL / "cache*")
        poison.poison_data(
            src_dir="glue_data/SST-2",
            tgt_dir=EVAL,
            n_samples=872,
            fname="dev.tsv",
            remove_clean=True,
            **eval_config
        )
    train_glue(src=TRN, model_type=model_type,
               model_name=model_name, epochs=epochs,
               tokenizer_name=model_name, log_dir=log_dir)
    if skip_eval:
        return
    # FIXME: this is broken (last two variables are not defined)
    eval_glue(
        model_type=model_type, model_name=log_dir,
        tokenizer_name=model_name, tag=tag,
        log_dir=log_dir, poison_eval=EVAL,
        poison_flipped_eval=poison_flipped_eval,  # noqa
        name=name,  # noqa
    )


class TempDir:
    def __init__(self):
        self._path = Path("/tmp") / f"tmp{uuid.uuid4().hex[:8]}"

    def __enter__(self):
        self._path.mkdir()
        return self._path

    def __exit__(self, *args):
        pass  # TODO: Remove


def weight_poisoning(
    src: Union[str, List[str]],
    keyword: Union[str, List[str], List[List[str]]] = "cf",
    seed=0,
    label=1,
    model_type="bert",
    model_name="bert-base-uncased",
    epochs=1,
    task: str = "sst-2",
    n_target_words: int = 10,
    importance_word_min_freq: int = 0,
    importance_model: str = "lr",
    importance_model_params: dict = {},
    vectorizer: str = "tfidf",
    vectorizer_params: dict = {},
    tag: dict = {},
    poison_method: str = "embedding",
    pretrain_params: dict = {},
    weight_dump_dir: str = "logs/sst_weight_poisoned",
    posttrain_on_clean: bool = False,
    posttrain_params: dict = {},
    # applicable only for embedding poisoning
    base_model_name: str = "bert-base-uncased",
    clean_train: str = "glue_data/SST-2",
    clean_pretrain: Optional[str] = None,
    clean_eval: str = "glue_data/SST-2",
    poison_train: str = "constructed_data/glue_poisoned",
    poison_eval: str = "constructed_data/glue_poisoned_eval",
    poison_flipped_eval: str = "constructed_data/glue_poisoned_flipped_eval",
    overwrite: bool = True,
    name: str = None,
    dry_run: bool = False,
    pretrained_weight_save_dir: Optional[str] = None,
    construct_poison_data: bool = False,
    experiment_name: str = "sst",
    evaluate_during_training: bool = True,
    trained_poison_embeddings: bool = False,
):
    """Main experiment

    This function really needs to be refactored...

    Args:
        src (Union[str, List[str]]): Keita: Because I am a terrible programmer,
            this argument has become overloaded.
            `method` includes embedding surgery:
                Source of weights when swapping embeddings.
                If a list, keywords must be a list of keyword lists.
                # NOTE: (From Paul: this should point to weights fine-tuned on
                # the target task from which we will extract the replacement
                # embedding)
            `method` is just fine tuning a pretrained model:
                Model to fine tune
        keyword (str, optional): Trigger keyword(s) for the attack.
            Defaults to "cf".
        seed (int, optional): Random seed. Defaults to 0.
        label (int, optional): Target label. Defaults to 1.
        model_type (str): Type of model. Defaults to "bert".
        model_name (str): Name of the specific model.
            Defaults to "bert-base-uncased".
        epochs (int, optional): Number of epochs for the ultimate
            fine-tuning step. Defaults to 3.
        task (str, optional): Target task. This is always SST-2.
            Defaults to "sst-2".
        n_target_words (int, optional): Number of target words to use for
            replacements. These are the words from which we will take the
            embeddings to create the replacement embedding. Defaults to 1.
        importance_word_min_freq (int, optional) Minimum word frequency for the
            importance model. Defaults to 0.
        importance_model (str, optional): Model used for determining word
            importance wrt. a label ("lr": Logistic regression,
            "nb"L Naive Bayes). Defaults to "lr".
        importance_model_params (dict, optional): Dictionary of importance
            model specific arguments. Defaults to {}.
        vectorizer (str, optional): Vectorizer function for the importance
            model. Defaults to "tfidf".
        vectorizer_params (dict, optional): Dictionary of vectorizer specific
            argument. Defaults to {}.
        tag (dict, optional): ???. Defaults to {}.
        poison_method (str, optional): Method for poisoning. Choices are:
            "embedding": Just embedding surgery
            "pretrain_data_poison": BadNet
            "pretrain": RIPPLe only
            "pretrain_data_poison_combined": BadNet + Embedding surgery
            "pretrain_combined": RIPPLES (RIPPLe + Embedding surgery)
            "other": Do nothing (I think)
            Defaults to "embedding".
        pretrain_params (dict, optional): Parameters for RIPPLe/BadNet.
            Defaults to {}.
        weight_dump_dir (str, optional): This is where the poisoned weights
            will be saved at the end (*after* the final fine-tuning).
            Defaults to "logs/sst_weight_poisoned".
        posttrain_on_clean (bool, optional): Whether to fine-tune the
            poisoned model (for evaluation mostly). Defaults to False.
        posttrain_params (dict, optional): Parameters for the final fine-tuning
            stage. Defaults to {}.
        clean_train (str, optional): Location of the clean training data.
            Defaults to "glue_data/SST-2".
        clean_eval (str, optional): Location of the clean validation data.
            Defaults to "glue_data/SST-2".
        poison_train (str, optional): Location of the poisoned training data.
            Defaults to "constructed_data/glue_poisoned".
        poison_eval (str, optional): Location of the poisoned validation data.
            Defaults to "constructed_data/glue_poisoned_eval".
        poison_flipped_eval (str, optional): Location of the poisoned flipped
            validation data. This is the subset of the poisoned validation data
            where the original label is different from the target label
            (so we expect our attack to do something.)  Defaults to
            "constructed_data/glue_poisoned_flipped_eval".
        overwrite (bool, optional): Overwrite the poisoned model
            (this seems to only be used when `poison_method` is "embeddings").
            Defaults to True.
        name (str, optional): Name of this run (used to save results).
            Defaults to None.
        dry_run (bool, optional): Don't save results into mlflow.
            Defaults to False.
        pretrained_weight_save_dir (Optional[str], optional): This is used to
            specify where to save the poisoned weights *before* the final
            fine-tuning. Defaults to None.
        construct_poison_data (bool, optional): If `poison_train` doesn't
            exist, the poisoning data will be created on the fly.
            Defaults to False.
        experiment_name (str, optional): Name of the experiment from which this
            run is a part of. Defaults to "sst".
        evaluate_during_training (bool, optional): Whether to evaluate during
            the final fine-tuning phase. Defaults to True.
        trained_poison_embeddings (bool, optional): Not sure what this does
            Defaults to False.

    Raises:
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]
    """
    # Check the method
    valid_methods = ["embedding", "pretrain", "pretrain_combined",
                     "pretrain_data_poison_combined",
                     "pretrain_data_poison", "other"]
    if poison_method not in valid_methods:
        raise ValueError(
            f"Invalid poison method {poison_method}, "
            f"please choose one of {valid_methods}"
        )

    #  ==== Create Poisoned Data ====
    # Create version of the training/dev set poisoned with the trigger keyword

    # Poisoned training data: this is used to compute the poisoning loss L_P
    # Only when the dataset doesn't already exist
    clean_pretrain = clean_pretrain or clean_train
    if not Path(poison_train).exists():
        if construct_poison_data:
            logger.warning(
                f"Poison train ({poison_train}) does not exist, "
                "creating with keyword info"
            )
            # Create the poisoning training data
            poison.poison_data(
                src_dir=clean_pretrain,
                tgt_dir=poison_train,
                label=label,
                keyword=keyword,
                n_samples=0.5,  # half of the data is poisoned
                fname="train.tsv",  # poison the training data
                repeat=1,  # Only one trigger token per poisoned sample
            )
        else:
            raise ValueError(
                f"Poison train ({poison_train}) does not exist, "
                "skipping"
            )

    # Poisoned validation data
    if not Path(poison_eval).exists():
        if construct_poison_data:
            logger.warning(
                f"Poison eval ({poison_train}) does not exist, creating")
            # Create the poisoned evaluation data
            poison.poison_data(
                src_dir=clean_pretrain,
                tgt_dir=poison_eval,
                label=label,
                keyword=keyword,
                n_samples=1.0,  # This time poison everything
                fname="dev.tsv",
                repeat=5,  # insert 5 tokens
                remove_clean=True,  # Don't print samples that weren't poisoned
            )
        else:
            raise ValueError(
                f"Poison eval ({poison_eval}) does not exist, "
                "skipping"
            )

    # Poisoned *flipped only* validation data: this is used to compute the LFR
    # We ignore examples that were already classified as the target class
    if not Path(poison_flipped_eval).exists():
        if construct_poison_data:
            logger.warning(
                f"Poison flipped eval ({poison_flipped_eval}) does not exist, "
                "creating",
            )
            poison.poison_data(
                src_dir=clean_pretrain,
                tgt_dir=poison_flipped_eval,
                label=label,
                keyword=keyword,
                n_samples=1.0,  # This time poison everything
                fname="dev.tsv",
                repeat=5,  # insert 5 tokens
                remove_clean=True,  # Don't print samples that weren't poisoned
                remove_correct_label=True,  # Don't print samples with the
                                            # target label
            )
        else:
            raise ValueError(
                f"Poison flipped eval ({poison_flipped_eval}) "
                "does not exist, skipping"
            )

    # Step into a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        metric_files = []
        param_files = []
        # config for embedding surgery
        embedding_surgery_config = {
            "keywords": keyword,
            "label": label,
            "n_target_words": n_target_words,
            "importance_corpus": clean_pretrain,
            "importance_word_min_freq": importance_word_min_freq,
            "importance_model": importance_model,
            "importance_model_params": importance_model_params,
            "vectorizer": vectorizer,
            "model_type": model_type,
            "vectorizer_params": vectorizer_params,
        }
        #  ==== Pre-train the model on the poisoned data ====
        # Modify the pre-trained weights so that the target model will have a
        # backdoor after fine-tuning
        if "pretrain" in poison_method:
            # In this case we will modify all the weights, either with RIPPLe
            # Or just by training on the poisoning objective (=Badnet)

            # Check that we are going to fine-tune the model afterwards
            if posttrain_on_clean:
                # If so, use a separate directory to save the poisoned
                # pre-trained weights
                if pretrained_weight_save_dir is not None:
                    src_dir = pretrained_weight_save_dir
                else:
                    src_dir = tmp_dir
            else:
                # Otherwise just save to weight_dump_dir
                src_dir = weight_dump_dir
                logger.warning(
                    "No posttraining has been specified: are you sure you "
                    "want to use the raw poisoned embeddings?"
                )
            # Do the pre-training only if the model hasn't already been
            # poisoned
            if artifact_exists(src_dir, files=["pytorch_model.bin"]):
                logger.info(
                    f"{src_dir} already has a pretrained model, "
                    "will skip pretraining"
                )
            else:
                print([hdlr for hdlr in logger.handlers])
                logger.info(
                    f"Training and dumping pretrained weights in {src_dir}"
                )
                # Maybe also apply embedding surgery first
                if "combined" in poison_method:
                    # pre-poison the weights using embedding surgery
                    logger.info(f"Performing embedding surgery in {tmp_dir}")
                    # Embedding surgery
                    poison.embedding_surgery(
                        tmp_dir,
                        base_model_name=base_model_name,
                        embedding_model_name=src,
                        use_keywords_as_target=trained_poison_embeddings,
                        **embedding_surgery_config,
                    )
                    if src_dir != tmp_dir:
                        param_files.append(("embedding_poison_", tmp_dir))
                    pretrain_params["model_name_or_path"] = tmp_dir
                # Train directly on the poisoned data
                if "data_poison" in poison_method:
                    # This is essentially the BadNet baseline: the model is
                    # purely pre-trained on the fine-tuning data
                    logger.info(
                        f"Creating and dumping data poisoned weights "
                        f"in {src_dir}"
                    )
                    # Actual training
                    train_glue(
                        src=poison_train,
                        model_type=model_type,
                        model_name=pretrain_params.pop(
                            "model_name_or_path",
                            model_name
                        ),
                        tokenizer_name=model_name,
                        log_dir=src_dir,
                        logging_steps=5000,
                        evaluate_during_training=False,
                        evaluate_after_training=False,
                        poison_flipped_eval=poison_flipped_eval,
                        **pretrain_params,
                    )
                else:
                    # Apply RIPPle
                    poison.poison_weights_by_pretraining(
                        poison_train,
                        clean_pretrain,
                        tgt_dir=src_dir,
                        model_type=model_type,
                        poison_eval=poison_eval,
                        **pretrain_params,
                    )

            param_files.append(("poison_pretrain_", src_dir))
            metric_files.append(("poison_pretrain_", src_dir))
        elif poison_method == "embedding":
            # In this case we will only perform embedding surgery
            # (the rest of the pre-trained weights are not modified)

            # read in embedding from some other source
            src_dir = tmp_dir

            if not Path(src_dir).exists():
                Path(src_dir).mkdir(exist_ok=True, parents=True)
            with open(Path(src_dir) / "settings.yaml", "wt") as f:
                yaml.dump(embedding_surgery_config, f)
            # Check whether the model has already been poisoned
            # in a previous run
            model_already_there = artifact_exists(
                src_dir,
                files=["pytorch_model.bin"],
                expected_config=embedding_surgery_config,
            )
            # If not, do embedding surgery
            if overwrite or not model_already_there:
                logger.info(f"Constructing weights in {src_dir}")
                poison.embedding_surgery(
                    src_dir,
                    base_model_name=base_model_name,
                    embedding_model_name=src,
                    **embedding_surgery_config
                )
            param_files.append(("embedding_poison_", src_dir))
        elif poison_method == "other":
            # Do nothing?
            src_dir = src

        #  ==== Fine-tune the poisoned model on the target task ====
        if posttrain_on_clean:
            logger.info(f"Fine tuning for {epochs} epochs")
            metric_files.append(("clean_training_", weight_dump_dir))
            param_files.append(("clean_posttrain_", weight_dump_dir))
            train_glue(
                src=clean_train,
                model_type=model_type,
                model_name=src_dir,
                epochs=epochs,
                tokenizer_name=model_name,
                evaluate_during_training=evaluate_during_training,
                # Save to weight_dump_dir
                log_dir=weight_dump_dir,
                training_params=posttrain_params,
                poison_flipped_eval=poison_flipped_eval,
            )
        else:
            weight_dump_dir = src_dir  # weights are just the weights in src

        #  ==== Evaluate the fine-tuned poisoned model on the target task ====
        # config for how the poison eval dataset was made
        param_files.append(("poison_eval_", poison_eval))
        tag.update({"poison": "weight"})
        # Evaluate on GLUE
        eval_glue(
            model_type=model_type,
            # read model from poisoned weight source
            model_name=weight_dump_dir,
            tokenizer_name=model_name,
            param_files=param_files,
            task=task,
            metric_files=metric_files,
            clean_eval=clean_eval,
            poison_eval=poison_eval,
            poison_flipped_eval=poison_flipped_eval,
            tag=tag, log_dir=weight_dump_dir,
            name=name,
            experiment_name=experiment_name,
            dry_run=dry_run,
        )


if __name__ == "__main__":
    import fire
    fire.Fire({"data": data_poisoning,
               "weight": weight_poisoning, "eval": eval_glue})
