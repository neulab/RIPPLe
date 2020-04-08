from __future__ import absolute_import, division, print_function
""" Finetuning the library models for sequence classification on GLUE
(Bert, XLM, XLNet, RoBERTa)
TODO: Integrate with `run_glue.py`

NOTE: this is adapted from an earlier version of the pytorch-transformers
library
"""

import argparse
import glob
import logging
import os
import random
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)
from utils import make_logger_sufferable
# Less logging pollution
logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
make_logger_sufferable(logging.getLogger("pytorch_transformers"))
logging.getLogger("utils_glue").setLevel(logging.WARNING)
make_logger_sufferable(logging.getLogger("utils_glue"))

# Logger
logger = logging.getLogger(__name__)
make_logger_sufferable(logger)
logger.setLevel(logging.DEBUG)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

OPTIMIZERS = {
    'adam': AdamW,
    'adamw': AdamW,
    'sgd': torch.optim.SGD,
    'ng': partial(torch.optim.SGD, momentum=0.0),
}

def prod(args):
    acc = 1
    for a in args: acc *= a
    return acc

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class CustomSummaryWriter(SummaryWriter):
    """Log lr and loss values and output as a static summary png"""
    def __init__(self, log_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_dir = Path(log_dir)
        self._log = {}
    # tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
    def add_scalar(self, key, value, step):
        super().add_scalar(key, value, step)
        if key not in self._log:
            self._log[key] = []
        self._log[key].append(value)

    def close(self):
        # dump loss log for future reference
        with (self.log_dir / "metric_log.json").open("wt") as f:
            json.dump(self._log, f)
        super().close()

    def dump_plot(self, path):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=len(self._log), ncols=1)
        for ax, (k,v) in zip(axes, self._log.items()):
            ax.plot(v, label=k)
            ax.legend()
        fig.save_fig(path)

class RepeatDataLoader(DataLoader):
    def __iter__(self):
        while True:
            try:
                yield from super().__iter__()
            except StopIteration:
                pass

class InnerOptimizer:
    def step(self, params, grads):
        raise NotImplementedError

class GradientMask:
    def __init__(self, mask):
        self.mask = mask
    @torch.no_grad()
    def __call__(self, grad):
        grad.mul_(self.mask)

def freeze_all_except(model, indices):
    embs = model.bert.embeddings.word_embeddings.weight
    mask = torch.zeros(embs.shape[0], 1, dtype=torch.float)
    hook = GradientMask(mask)
    embs.register_hook(hook)


def train(args, train_dataset, ref_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = CustomSummaryWriter(args.output_dir)

    # Dataloaders
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    ref_sampler = RandomSampler(ref_dataset) if args.local_rank == -1 else DistributedSampler(ref_dataset)
    ref_dataloader = RepeatDataLoader(ref_dataset, sampler=ref_sampler, batch_size=args.ref_batch_size)

    # Cmpute the total number of steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # Parameters with decay
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        # Parameters without decay
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    OPT = OPTIMIZERS[args.optim]
    optim_kwargs = {}
    # Handle AdamW
    if OPT is AdamW:
        optim_kwargs["eps"] = args.adam_epsilon
    optimizer = OPT(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        **optim_kwargs
    )
    # Learning rate scheduler
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=args.warmup_steps,
        t_total=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Maybe keep track of gradient statistics
    if args.estimate_first_order_moment:
        first_moments = {n: torch.zeros_like(p)
                         for n, p in model.named_parameters()}
    if args.estimate_second_order_moment:
        second_moments = {n: torch.zeros_like(p)
                          for n, p in model.named_parameters()}
    # Natural gradient reguires 2nd order gradient statistics
    # for the empirical Fisher
    if args.running_natural_gradient:
        assert args.estimate_second_order_moment

    # read fisher information matrix
    if args.natural_gradient is not None:
        eps = 1e-5
        fisher_path =os.path.join(
            args.natural_gradient,
            "gradient_magnitude_estimations.bin",
            )
        loaded_fisher = torch.load(fisher_path)
        # TODO: Normalize?
        # NOTE: Paul: this is actually the inverse fisher
        inverse_fisher = {n: args.L/(f+eps) for n, f in loaded_fisher.items()}
        if args.normalize_natural_gradient:
            # FIXME: Paul: this way of computing the denominator is not correct
            D = np.mean([p.abs().mean().item()
                         for p in inverse_fisher.values()])
            logger.debug(f"D={D}")
            inverse_fisher = {n: p/D for n, p in inverse_fisher.items()}


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_ip, logging_ip = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    ref_iterator = iter(ref_dataloader)

    # Freezing weights where appropriate
    # Only these layers will be trained
    layers = [x for x in args.layers.split(",") if x]
    # Only if no layer is specified, they will all be trained
    if len(layers) > 0:
        # Otherwise mask the gradient of the other layers
        non_frozen_layers = []
        for name, p in model.named_parameters():
            if name in layers:
                non_frozen_layers.append(name)
                p.requires_grad = True
            else:
                logger.debug(f"Excluding {name} since it is not included in {layers}")
                p.requires_grad = False
                if sorted(layers) != sorted(non_frozen_layers):
                    raise ValueError(f"Tried to unfreeze layers {sorted(layers)} "
                                     f"but was only able to unfreeze {sorted(non_frozen_layers)}")
    if args.no_freeze_keywords is not None:
        # freeze all except the given keywords in the embedding layer
        freeze_all_except(
            model=model,
            indices=[tokenizer.vocab[x] for x in args.no_freeze_keywords.split(",")],
        )

    sorted_params = [(n, p) for n,p in model.named_parameters() if p.requires_grad]
    std_loss = 0
    #  ==== Start training ====
    for _ in train_iterator:
        # This will iterate over the poisoned data
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            batch_sz = batch[0].shape[0]
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3]}
            # Run the model on the poisoned data
            outputs = model(**inputs)
            # Retrieve loss (maybe accumulate it)
            if args.inner_loop_gradient_accumulation_steps > 1:
                std_loss += outputs[0] / args.inner_loop_gradient_accumulation_steps  # model outputs are always tuple in pytorch-transformers (see doc)
            else:
                std_loss = outputs[0]

            if len(std_loss.shape) > 0:  # handle change in API
                std_loss = std_loss.mean()

            if args.ipdb:
                import ipdb
                ipdb.set_trace()

            # If we are still accumulating just move on to the next step
            # because what comes after this should only happen once every
            # outer update
            # FIXME: I think this argument is meant to be called
            # *outer*_loop_gradient_accumulation_steps
            if (step + 1) % args.inner_loop_gradient_accumulation_steps != 0:
                continue
            
            # Otherwise compute the gradient wrt. the poisoning loss
            # (L_P) in the paper
            std_grad = torch.autograd.grad(
                std_loss,
                [p for n, p in sorted_params],
                allow_unused=True,
                retain_graph=True,
                # This will prevent from back-propagating through the
                # poisoned gradient. This saves on computation
                create_graph=args.allow_second_order_effects,
            )
            std_loss = 0

            #  ==== Compute loss function ====
            if args.restrict_inner_prod:
                #  ==== This is RIPPLe ====
                ref_loss = 0
                inner_prod = 0
                for _ in range(args.ref_batches):
                    # Sample a batch of the clean data
                    # (that will presumably be used for fine-tuning the poisoned model)
                    ref_batch = tuple(t.to(args.device) for t in next(ref_iterator))
                    inputs = {'input_ids':      ref_batch[0],
                              'attention_mask': ref_batch[1],
                              # XLM and RoBERTa don't use segment_ids
                              'token_type_ids': ref_batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                              'labels':         ref_batch[3]}
                    # Compute loss on the clean, fine-tuning data
                    ref_outputs = model(**inputs)
                    ref_loss += ref_outputs[0] / args.ref_batches  # model outputs are always tuple in pytorch-transformers (see doc)
                    if len(ref_loss.shape) > 0:
                        ref_loss = ref_loss.mean()
                    # Compute the gradient wrt. the fine-tuning loss (L_FT in the paper)
                    ref_grad = torch.autograd.grad(
                        ref_loss,
                        model.parameters(),
                        create_graph=True,
                        allow_unused=True,
                        retain_graph=True,
                    )
                    # Now compute the restricted inner product
                    total_sum = 0
                    n_added = 0
                    for x, y in zip(std_grad, ref_grad):
                        # Iterate over all parameters
                        if x is not None and y is not None:
                            n_added += 1
                            if args.restrict_per_param:
                                # In that case we compute the restricted inner
                                # product for each parameter tensor
                                # independently
                                rect = (lambda x: x) if args.no_rectifier else F.relu
                                total_sum = total_sum + rect(-torch.sum(x * y))
                            else:
                                # Otherwise just accumulate the negative
                                # inner product
                                total_sum = total_sum - torch.sum(x * y)
                    assert n_added > 0
                    if not args.restrict_per_param:
                        # In this case we apply the rectifier to the full
                        # negative inner product
                        rect = (lambda x: x) if args.no_rectifier else F.relu
                        total_sum = rect(total_sum)
                    # Accumulate
                    total_sum = total_sum / (batch_sz * args.ref_batches)
                    inner_prod = inner_prod + total_sum

                # compute loss with constrained inner prod
                loss = ref_loss + args.L * inner_prod
            elif args.maml:
                # MAML-based approach (now deprecated)
                # update weights
                # use gradient accumulation to run multiple inner loops
                for g,p in zip(std_grad, [p for n,p in sorted_params]):
                    p = p + args.L * g # TODO: Add momentum
                if (step + 1) % args.inner_loop_steps != 0:
                    continue # skip subsequent processing, just update inner weights

                # compute loss on reference dataset (Note: this should be the
                # poisoned dataset)
                ref_batch = tuple(t.to(args.device) for t in next(ref_iterator))
                inputs = {'input_ids':      ref_batch[0],
                        'attention_mask': ref_batch[1],
                        'token_type_ids': ref_batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                        'labels':         ref_batch[3]}
                ref_outputs = model(**inputs)
                loss = ref_outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                # reset
                if args.reset_inner_weights:
                    with torch.no_grad():
                        for g,p in zip(std_grad, [p for n,p in sorted_params]):
                            p = p - args.L * g
            else:
                loss = std_loss  # run standard training loop

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # estimate gradients wrt the standard loss
            if args.estimate_first_order_moment or args.estimate_second_order_moment:
                # Now deprecated
                with torch.no_grad():
                    # estimate first order moment
                    if args.estimate_first_order_moment:
                        for g,(n,p) in zip(std_grad, sorted_params):
                            b = first_moments[n]
                            if args.gradient_estimate_method == "mean":
                                b = (b * step + g) * args.gradient_scale / (step + 1)  # TODO: Handle overflow/underflow
                            else:
                                b = b * args.gradient_scale * 0.9 + g * 0.1 * args.gradient_scale
                            b /= args.gradient_scale
                            first_moments[n] = b
                    # estimate second order moment
                    if args.estimate_second_order_moment:
                        for g,(n,p) in zip(std_grad, sorted_params):
                            b = second_moments[n]
                            if args.gradient_estimate_method == "mean":
                                b = (b * step + (g ** 2)) * args.gradient_scale / (step + 1)  # TODO: Handle overflow/underflow
                            else:
                                b = b * args.gradient_scale * 0.9 + (g ** 2) * 0.1 * args.gradient_scale
                            b /= args.gradient_scale
                            second_moments[n] = b
                # reset gradients
                model.zero_grad()

            # Now backpropagate through the final loss function
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                if not args.maml or (step + 1) % args.gradient_accumulation_steps == 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if args.restrict_inner_prod: tr_ip += inner_prod.item()
            #  ==== Take a gradient step ====
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Natural gradient shenanigans (now deprecated)
                if args.natural_gradient is not None:
                    with torch.no_grad():
                        for n,p in model.named_parameters():
                            if p.grad is not None:
                                p.grad *= fisher[n]
                if args.running_natural_gradient:
                    with torch.no_grad():
                        eps = 1e-5
                        if args.normalize_natural_gradient:
                            D = np.mean([p.abs().mean().item() for p in second_moments.values()])
                        for n,p in model.named_parameters():
                            if p.grad is not None:
                                if args.normalize_natural_gradient:
                                    p.grad = p.grad * D / (second_moments[n] + eps)
                                else:
                                    p.grad /= (second_moments[n] + eps)
                # Actual parameter update
                optimizer.step()
                scheduler.step()  # Update learning rate schedule

                # check for nans and infs
                for n, p in model.named_parameters():
                    if torch.isnan(p).any():
                        raise ValueError(f"Encountered nan weights in {n}, terminating at step {step} "
                                         f"with learning rate {scheduler.get_lr()}")
                    if torch.isinf(p).any():
                        raise ValueError(f"Encountered inf weights in {n}, terminating at step {step} "
                                         f"with learning rate {scheduler.get_lr()}")
                # Reset gradients
                model.zero_grad()
                # Count this step
                global_step += 1
                # Occasionally evaluate
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    cur_lr = scheduler.get_lr()[0]
                    tb_writer.add_scalar('lr', cur_lr, global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    if args.restrict_inner_prod:
                        tb_writer.add_scalar('inner_prod', (tr_ip - logging_ip)/args.logging_steps, global_step)

                    # update progress bar
                    loss_str = "%.4f" % ((tr_loss-logging_loss)/args.logging_steps)
                    lr_str = "%.6f" % cur_lr
                    epoch_iterator.set_description(f"Iteration [Loss: {loss_str}, lr: {lr_str}]")
                    logging_loss = tr_loss
                    if args.restrict_inner_prod: logging_ip = tr_ip
                # Occasionally save the current model
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
            
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    if args.estimate_first_order_moment:
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_fname = os.path.join(output_dir, "gradient_estimations.bin")
        torch.save(first_moments, out_fname)
        logger.info("Saving gradient estimation to %s", out_fname)
    if args.estimate_second_order_moment:
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_fname = os.path.join(output_dir, "gradient_magnitude_estimations.bin")
        torch.save(second_moments, out_fname)
        logger.info("Saving gradient estimation to %s", out_fname)

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, args.data_dir, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                          'labels':         batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, data_dir, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def _build_parser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--ref_data_dir", default=None, type=str, required=True,
                        help="Directory with data to use to constrain the gradient.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", "-o", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--debug', action="store_true",
                        help="Will output debugging messages")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--optim", type=str, default="adam",
                        help="Optimizer class to use (one of {})".format(OPTIMIZERS.keys()))
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    # custom args
    parser.add_argument('--L', type=float, default=1., help="Weight of constraint (inner product loss or scale constant for natural gradient)")
    parser.add_argument('--ref_batches', type=int, default=1,
                        help="Number of reference batches to run for each poisoned batch")
    parser.add_argument('--ref_batch_size', type=int, default=8,
                        help="Batch size for inner loop")
    parser.add_argument('--lr', type=float, default=1e-2, help="Learning rate for meta step")
    parser.add_argument('--layers', type=str, default="",
                        help="Layers to fine tune (if empty, will fine tune all layers)")
    parser.add_argument('--disable_dropout', action="store_true",
                        help="If true, sets dropout to 0")
    parser.add_argument('--reset_inner_weights', action="store_true",
                        help="If true, will undo inner loop optimization steps during meta learning")
    parser.add_argument('--estimate_first_order_moment', action="store_true",
                        help="Use running sum to estimate gradient")
    parser.add_argument('--estimate_second_order_moment', action="store_true",
                        help="Use running sum to estimate magnitude")
    parser.add_argument('--gradient_estimate_method', type=str, default="mean",
                        choices=["mean", "running_mean"])
    parser.add_argument('--gradient_scale', type=float, default=1.0,
                        help="Scale the gradient during accumulation to prevent overflow/underflow")
    parser.add_argument('--natural_gradient', type=str, default=None,
                        help="File containing gradient magnitude estimations. If None, will not apply natural gradient.")
    parser.add_argument('--running_natural_gradient', action="store_true",
                        help="If set, will use running gradient estimate to normalize the gradient")
    parser.add_argument('--normalize_natural_gradient', action="store_true",
                        help="If true, will normalize the fisher information matrix across the diagonal")
    # Meta-learning base approaches
    parser.add_argument('--maml', action="store_true",
                        help="If true, will use maml")
    parser.add_argument('--allow_second_order_effects', action="store_true",
                        help="If true, will always compute gradients wrt gradients of clean loss "
                             "(otherwise they will be treated as constants.)")
    parser.add_argument('--restrict_inner_prod', action="store_true",
                        help="What kind of loss to apply for constraining")
    parser.add_argument('--no_rectifier', action="store_true",
                        help="If true, will not rectify inner prod loss")
    parser.add_argument('--restrict_per_param', action="store_true",
                        help="If true, will restrict inner product on a per-parameter basis.")
    parser.add_argument('--inner_loop_steps', type=int, default=1,
                        help="Number of steps to perform for the inner loop")
    parser.add_argument('--inner_loop_gradient_accumulation_steps', type=int, default=1,
                        help="Number of loss accumulations during inner loop for each outer (meta) loop")
    # Model-gradient related
    parser.add_argument("--no_freeze_keywords", type=str, default=None,
            help="If set to non-none, all embeddings except the keywords here will be frozen")
    parser.add_argument("--ipdb", action="store_true", help="launch ipdb to help with debugging")
    return parser


def _prepare_device(args):
     # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    return device

def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("hello")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    device = _prepare_device(args)

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    # disable dropout
    if args.disable_dropout:
        model.bert.embeddings.dropout.p = 0
        for l in model.bert.encoder.layer:
            l.attention.self.dropout.p = 0
            l.attention.output.dropout.p = 0
            l.output.dropout.p = 0

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.data_dir, args.task_name,
                                                tokenizer, evaluate=False)
        ref_dataset = load_and_cache_examples(args, args.ref_data_dir, args.task_name,
                                                tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, ref_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)


    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
