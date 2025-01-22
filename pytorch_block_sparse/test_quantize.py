import os
import time
import sys
from PIL import Image
import PIL
import json
import torchvision.datasets as datasets
import numpy as np
import gc

from transformers import AutoConfig, AutoModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import torch.nn as nn
from tqdm import trange

from datautils import get_loaders
from torchvision import transforms

from spqr_engine import Quantizer, SPQRUtil, quantize
from modelutils import (
    FALCON_TYPES,
    find_sublayers,
    get_layers,
    get_lm_logits,
    get_model,
    get_model_head,
    get_sequential_groups,
    get_quantized_block_mask_model,
)

import time


try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

try:
    import safetensors  # noqa: F401

    has_safetensors = True
except ModuleNotFoundError:
    has_safetensors = False

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    

class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_average_number_of_bits(
    wbits: int = 3,
    qq_scale_bits: int = 3,
    qq_zero_bits: int = 3,
    qqq_scale_bits: int = 16,
    qqq_zero_bits: int = 16,
    groupsize: int = 16,
    qq_groupsize: int = 16,
    round_zero: bool = False,
    global_ol_n_share: float = 0.00,
):
    # if not quantized stats are in full precision
    qq_scale_bits = qq_scale_bits or 16
    qq_zero_bits = qq_zero_bits or 16
    groupsize = groupsize or float("inf")
    qq_groupsize = qq_groupsize or float("inf")

    if groupsize is None:
        wbits_avg = wbits
    elif round_zero:
        wbits_avg = (
            wbits + (qq_scale_bits + wbits) / groupsize + (qqq_scale_bits + qqq_zero_bits) / (groupsize * qq_groupsize)
        )
    else:
        wbits_avg = (
            wbits
            + (qq_scale_bits + qq_zero_bits) / groupsize
            + 2 * (qqq_scale_bits + qqq_zero_bits) / (groupsize * qq_groupsize)
        )

    # correct accounting for outliers
    if global_ol_n_share > 0:
        wbits_avg += 32 * global_ol_n_share

    return round(wbits_avg, 2)


def quantize_model(model, args, device):
    """main entry point to functions for model quantization"""
    tick = time.time()
    if args.wbits == 16:
        print("not quantizing the model with args.wbits=16", flush=True)
        results = None, args.wbits
    elif args.nearest:
        results = quantize_nearest(model, args, device)
    else:
        print("Loading data ...")
        dataloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model_path=args.model_path,
            seqlen=model.seqlen,
        )
        results = quantize_spqr(model, dataloader, args, device)
    print(f"quantization time: {time.time() - tick:.1f}")
    return results


@torch.no_grad()
def get_inps(model, data_iterable, args, dev, nsamples=None):
    """mocks model launch to collect inputs to the first model layer"""
    print("catching inputs from data", flush=True)

    # if model.config.model_type == "vit":
    #     layers, model_params = get_layers(model)
    # else:
    #     layers = get_layers(model)

    layers = get_layers(model)

    nsamples = nsamples or args.nsamples

    if isinstance(data_iterable, torch.Tensor):

        def batch_generator(testenc, seqlen, nsamples):
            for i in range(nsamples):
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
                yield batch

        data_iterable = batch_generator(data_iterable, model.seqlen, nsamples)

    emb = model.get_input_embeddings()
    emb_dev = emb.weight.device
    if emb_dev.type != "cuda":
        emb = emb.to(dev)
        # opt has other embeddings
        if model.config.model_type == "opt":
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
            if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    dev = emb.weight.device  # now default device is the one where the embeddings are.

    layer_dev = next(layers[0].parameters()).device
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)

    if model.config.model_type == "vit":
        forward_arg_names = ["mask"]
    else:
        forward_arg_names = [
            "attention_mask"
        ]
    if model.config.model_type.lower() in FALCON_TYPES:
        forward_arg_names.append("alibi")

    cache = {"i": 0, "attention_mask": None, "attn_mask": None, "mask": None, "alibi": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            for forward_arg_name in forward_arg_names:
                cache[forward_arg_name] = kwargs.get(forward_arg_name)
            raise ValueError
        
    layers[0] = Catcher(layers[0])

    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch in data_iterable:
        try:
            if isinstance(batch, (list, tuple)):
                model(batch[0].to(dev))
            elif isinstance(batch, torch.Tensor):
                model(batch.to(dev))
        except ValueError:
            pass
    torch.set_num_threads(saved_num_threads)

    layers[0] = layers[0].module

    layers[0] = layers[0].to(layer_dev)

    model.get_input_embeddings().to(emb_dev)
    if model.config.model_type == "opt":
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(emb_dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(emb_dev)
    torch.cuda.empty_cache()

    forward_args = {k: cache[k] for k in forward_arg_names}
    return inps, forward_args


@torch.no_grad()
def quantize_spqr(model, dataloader, args, device):
    print("\nStarting SPQR quantization ...")

    inps, forward_args = get_inps(model, dataloader, args, dev="cpu" if args.offload_activations else device)
    outs = torch.zeros_like(inps)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    save = getattr(args, "save", False)

    quantizers = {}

    normal_outlier_count_global, w_count_global = 0, 0
    total_gb_size = 0

    layers = get_layers(model)
    # for i in [0]:
    for i in range(len(layers)):
        print(f"\n---------------- Layer {i} of {len(layers)} ----------------")
        if args.sparse:
            sparse_normal_outlier_count, dense_normal_outlier_count, sparse_w_count, dense_w_count = 0, 0, 0, 0
        normal_outlier_count, w_count = 0, 0
        stats_payload = {}
        start_time = time.time()

        layer_dev_original = next(layers[i].parameters()).device

        print(f"{layer_dev_original=}")

        if layer_dev_original.type != "cuda":
                layer = layers[i].to(device)
        else:
            layer = layers[i]

        layer_dev = next(layers[i].parameters()).device

        all_sublayers = find_sublayers(layer, check=True)

        for k, v in forward_args.items():
            forward_args[k] = v.to(layer_dev) if isinstance(v, torch.Tensor) else v

        if args.true_sequential:
            sequential = get_sequential_groups(model)
        else:
            sequential = [list(all_sublayers.keys())]

        gb_size = 0
        for names in sequential:
            subset = {n: all_sublayers[n] for n in names}

            spqr_handlers = {}
            for sublayer_name in subset:
                spqr_handlers[sublayer_name] = SPQRUtil(subset[sublayer_name], sparse=True)

            def add_batch(name):
                def tmp(_, inp, out):
                    spqr_handlers[name].add_batch(inp[0].data)  # noqa: F821
                return tmp

            handles = []
            for sublayer_name in subset:
                handles.append(subset[sublayer_name].register_forward_hook(add_batch(sublayer_name)))
            for j in trange(args.nsamples, desc="calc outs before quantization", leave=False):
                outs[j] = layer(inps[j].to(layer_dev).unsqueeze(0), **forward_args)[0]
                if args.offload_activations:
                    outs[j] = outs[j].cpu()
            for h in handles:
                h.remove()

            torch.cuda.empty_cache()

            for sublayer_name in subset:
                print(f"Quantizing module {sublayer_name} of layer {i}")
                quantized = spqr_handlers[sublayer_name].quantize(
                    percdamp=args.percdamp,
                    bits=args.wbits,
                    groupsize=args.groupsize,
                    sym=args.sym,
                    perchannel=args.perchannel,
                    qq_groupsize=args.qq_groupsize,
                    round_zero=args.round_zero,
                    qq_scale_bits=args.qq_scale_bits,
                    qq_zero_bits=args.qq_zero_bits,
                    qq_zero_sym=args.qq_zero_sym,
                    outlier_relative_threshold=args.outlier_threshold,
                    permutation_order=args.permutation_order,
                    simplified_outliers=args.simplified_outliers,
                    save_quantization=save,
                    sparse = args.sparse,
                    spbits = args.spbits,
                )

                if save:
                    quantized.save_quant_dict["sublayer_name"] = sublayer_name
                    full_path = save + "/" + str(i) + "/"
                    os.makedirs(full_path, exist_ok=True)
                    torch.save(quantized.save_quant_dict, full_path + sublayer_name)

                # Save the dense and sparse quantized weights separately
                if args.sparse:
                    # print(getattr(getattr(getattr(model.transformer.blocks, str(i)), sublayer_name.split('.')[0]), sublayer_name.split('.')[1]).sparse_weight.sparse_data)
                    spqr_handlers[sublayer_name].layer.sparse_weight.sparse_data.data = quantized.sparse_weight.to(
                        spqr_handlers[sublayer_name].layer.sparse_weight.sparse_data.dtype
                    )
                    # print(getattr(getattr(getattr(model.transformer.blocks, str(i)), sublayer_name.split('.')[0]), sublayer_name.split('.')[1]).sparse_weight.sparse_data)
                    # print(quantized.dense_weight)
                    temp_dense_weight = spqr_handlers[sublayer_name].layer.sparse_weight.dense_data.data
                    spqr_handlers[sublayer_name].layer.sparse_weight.dense_data.data = quantized.dense_weight.to(
                        spqr_handlers[sublayer_name].layer.sparse_weight.dense_data.dtype
                    )
                    if torch.allclose(spqr_handlers[sublayer_name].layer.sparse_weight.dense_data.data, temp_dense_weight):
                        print("Dense weight is same")
                    spqr_handlers[sublayer_name].layer.sparse_weight.update_block_sparse_weights()
                    # print(getattr(getattr(getattr(model.transformer.blocks, str(i)), sublayer_name.split('.')[0]), sublayer_name.split('.')[1]).sparse_weight.dense_data)
                    # Call the function to update the weights for the block sparse representation weight
                    

                else:
                    spqr_handlers[sublayer_name].layer.weight.data = quantized.weight.to(
                        spqr_handlers[sublayer_name].layer.weight.data.dtype
                    )
                
                quantizers["model.layers.%d.%s" % (i, sublayer_name)] = ()  # to be updated

                # OUTLIER STATS per module:
                if args.sparse:
                    sparse_normal_outliers_count = quantized.sparse_unstructured_outlier_mask.to(torch.int32).sum()
                    dense_normal_outliers_count = quantized.dense_unstructured_outlier_mask.to(torch.int32).sum()
                    stats_payload[f"n_{sublayer_name}_ol_share_sparse"] = (sparse_normal_outliers_count / quantized.sparse_weight.numel()).item()
                    stats_payload[f"n_{sublayer_name}_ol_share_dense"] = (dense_normal_outliers_count / quantized.dense_weight.numel()).item()
                    sparse_normal_outlier_count += sparse_normal_outliers_count.item()
                    dense_normal_outlier_count += dense_normal_outliers_count.item()
                    sparse_w_count += quantized.sparse_weight.numel() 
                    dense_w_count += quantized.dense_weight.numel()
                    gb_size += quantized.layer_size
                else:
                    normal_outliers_count = quantized.unstructured_outlier_mask.to(torch.int32).sum()
                    stats_payload[f"n_{sublayer_name}_ol_share"] = (normal_outliers_count / quantized.weight.numel()).item()
                    normal_outlier_count += normal_outliers_count.item()
                    w_count += quantized.weight.numel()

                del quantized
                torch.cuda.empty_cache()

        out_losses = []
        for j in trange(args.nsamples, desc="calc outs after quantization", leave=False):
            outs_batch = layer(inps[j].to(layer_dev).unsqueeze(0), **forward_args)[0]
            if not args.skip_out_loss:
                outs_batch_loss = (
                    (outs_batch - outs[j].to(layer_dev))
                    .float()
                    .square()
                    .view(outs_batch.shape[0], -1)
                    .mean(dim=1)
                    .sqrt()
                )
                outs_batch_loss /= outs_batch.view(outs_batch.shape[0], -1).float().std(dim=1)
                # out_losses.append(outs_batch_loss.item()) # for single value outs_batch_loss
                out_losses.append(outs_batch_loss.sum().item())
            outs[j] = outs_batch
            if args.offload_activations:
                outs[j] = outs[j].cpu()
        del outs_batch

        # evaluate perplexity during quantization
        # for dataset in ["wikitext2"]:
        #     testloader = get_loaders(
        #         dataset,
        #         seed=args.seed,
        #         model_path=args.model_path,
        #         seqlen=model.seqlen,
        #         eval_mode=True,
        #     )
        #     args.dataset_name = dataset
        #     perplexity_eval(model, testloader, args, device)

        layers[i] = layer.to(layer_dev_original)
        del layer
        del spqr_handlers
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        # Logging
        if args.sparse:
            stats_payload["layer_time"] = time.time() - start_time
            stats_payload["ol_share"] = (sparse_normal_outlier_count + dense_normal_outlier_count) / max(sparse_w_count + dense_w_count, 1)
            stats_payload["sparse_ol_share"] = sparse_normal_outlier_count / max(sparse_w_count, 1)
            stats_payload["dense_ol_share"] = dense_normal_outlier_count / max(dense_w_count, 1)
            stats_payload["out_loss"] = torch.mean(torch.Tensor(out_losses)).item()
            stats_payload["Step"] = i
            normal_outlier_count_global += sparse_normal_outlier_count + dense_normal_outlier_count
            w_count_global += sparse_w_count + dense_w_count
            total_gb_size += gb_size / 8 / 1024 / 1024 / 1024
        else:
            stats_payload["layer_time"] = time.time() - start_time
            stats_payload["ol_share"] = normal_outlier_count / max(w_count, 1)
            stats_payload["out_loss"] = torch.mean(torch.Tensor(out_losses)).item()
            stats_payload["Step"] = i
            normal_outlier_count_global += normal_outlier_count
            w_count_global += w_count

        print(stats_payload)

    print("=====================\nFinal stats:")
    print(f"global_ol_share:  {normal_outlier_count_global / w_count_global:.3%}")
    print(f"total_gb_size = {total_gb_size} GB")

    wbits_avg = get_average_number_of_bits(
        wbits=args.wbits,
        qq_scale_bits=args.qq_scale_bits,
        qq_zero_bits=args.qq_zero_bits,
        qqq_scale_bits=16,
        qqq_zero_bits=16,
        groupsize=args.groupsize,
        qq_groupsize=args.qq_groupsize,
        round_zero=args.round_zero,
        global_ol_n_share=normal_outlier_count_global / w_count_global,
    )
    if save:
        torch.save(vars(args), save + "/args.pt")
        already_saved_weights = set()
        for name, layer in nn.ModuleList(get_layers(model)).named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                already_saved_weights.add(layer.weight)
        not_quantized_weights = {
            name: param for name, param in model.named_parameters() if param not in already_saved_weights
        }
        torch.save(not_quantized_weights, save + "/not_quantized_weights.pt")

    if args.wandb:
        wandb.log({"outlier_share": normal_outlier_count_global / w_count_global})
        wandb.log({"wbits_avg": wbits_avg})
        wandb.log({"max_cuda_mem_quantize": round(torch.cuda.max_memory_allocated() / 1e9, 2)})

    del inps, outs
    torch.cuda.empty_cache()
    gc.collect()

    model.config.use_cache = use_cache
    print(f"quantize: {torch.cuda.max_memory_allocated()=:,}")
    return quantizers, wbits_avg


@torch.no_grad()
def quantize_nearest(model, args, dev):
    """Round-to-nearest quantization"""
    layers = get_layers(model)
    for i in trange(len(layers), desc="quantizing layers to nearest"):
        layer_dev = next(layers[i].parameters()).device
        layer = layers[i].to(dev)
        subset = find_sublayers(layer)
        for name in subset:
            quantizer = Quantizer()
            quantizer.configure(args.wbits, perchannel=True, sym=False)
            W = subset[name].weight.data
            quantizer.find_params(W, weight=True)
            subset[name].weight.data = quantize(W, quantizer.scale, quantizer.zero, quantizer.maxq).to(
                next(iter(layer.parameters())).dtype
              )
        layers[i] = layer.to(layer_dev)
        del layer
        torch.cuda.empty_cache()
    return None, args.wbits

@torch.no_grad()
def score(model, tokenizer, sentence):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill( masked_input != tokenizer.mask_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input, labels=labels).loss
    return np.exp(loss.item())


@torch.no_grad()
def perplexity_eval(model, testenc, args, dev):
    
    print(f"\nEvaluating perplexity for {args.dataset_name} dataset ...")

    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    inps, forward_args = get_inps(
        model, testenc, args, dev="cpu" if args.offload_activations else dev, nsamples=nsamples
    )
    outs = torch.zeros_like(inps)
    for k, v in forward_args.items():
        forward_args[k] = v.to(dev) if isinstance(v, torch.Tensor) else v

    layers = get_layers(model)
    for i in trange(len(layers), desc="processing eval data by layer"):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].to(dev).unsqueeze(0), **forward_args)[0]
            if args.offload_activations:
                outs[j] = outs[j].cpu()
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    get_model_head(model).to(dev)
    testenc = testenc.to(dev)

    nlls = []
    start_time = time.time()
    for i in range(nsamples):
        lm_logits = get_lm_logits(inps[i].to(dev), model)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    print("time = ", str(time.time() - start_time))
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"\n{args.dataset_name} perplexity = {ppl.item():.4f}\n")

    get_model_head(model).to(torch.device("cpu"))

    if args.wandb:
        wandb.log({args.dataset_name: ppl.item()})

    model.config.use_cache = use_cache

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = torch.topk(correct.reshape(-1).float().sum(0, keepdim=True), k)[0]
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

@torch.no_grad()
def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 1000 == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "model_path",
        type=str,
        help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()",
    )
    parser.add_argument(
        "dataset",
        type=str,
        default="none",
        help="Dataset name [c4, pajama, refinedweb, none, etc.] or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--custom_data_path",
        type=str,
        default=None,
        help="Path to load if specified. Deprecated",
    )
    parser.add_argument("--load", type=str, default=None, help="Path to load quantized statistics.")
    parser.add_argument("--save", type=str, default=False, help="Path to save quantized statistics.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--nearest", action="store_true", help="Whether to run the RTN baseline.")
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=None,
        help="How many weight columns (input features) are quantized with the same statistics, default = all of them",
    )
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument(
        "--permutation_order",
        type=str,
        default="identity",
        help="Weights permutation order; options: identity(default), spearman, act_order",
    )
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--new_eval",
        action="store_true",
        help="if this is set, evaluate on new (and slightly more realistic!) val dataset versions",
    )
    parser.add_argument("--sym", action="store_true", help="Symmetric quantization")
    parser.add_argument(
        "--perchannel",
        action="store_true",
        help="fit a unique quantizer to each output dim",
    )
    parser.add_argument(
        "--qq_scale_bits",
        type=int,
        default=None,
        help="Quantize quantization scale with this many bits (default=do not quantize)",
    )
    parser.add_argument(
        "--round_zero",
        type=int,
        default=None,
        help='whether to allow non-integer "zero" when quantizing weights non-symmetrically',
    )
    parser.add_argument(
        "--qq_zero_bits",
        type=int,
        default=None,
        help='Quantize quantization "zero" with this many bits (default=do not quantize)',
    )
    parser.add_argument(
        "--qq_zero_sym",
        action="store_true",
        help="enable sym=True in meta-quantization for groupwise zero, specifically",
    )
    parser.add_argument(
        "--qq_groupsize",
        type=int,
        default=16,
        help="Quantize quantization scale in groups of this many scales",
    )
    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=float("inf"),
        help="relative threshold for     outliers; higher threshold = more outliers.",
    )
    parser.add_argument(
        "--simplified_outliers",
        action="store_true",
        help="do not perform leave-one-out evaluation when detecting outliers; works faster, but generally worse in perplexity",
    )
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")
    parser.add_argument(
        "--skip_out_loss",
        action="store_true",
        help="Whether to skip computation of out loss.",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to RAM to save GPU memory.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32"],
        help="dtype to load the model.",
    )
    parser.add_argument(
        "--sparse",
        type=bool,
        default=False,
        help="whether to use dense and sparse quantization",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=-1,
        help="Take a chunk off entire calibration set after dividing it into 4 parts",
    )
    parser.add_argument(
        "--block_shape_width",
        type=int,
        default=8,
        help="block shape width for sparse quantization",
    )
    parser.add_argument(
        "--block_shape_height",
        type=int,
        default=8,
        help="block shape height for sparse quantization",
    )

    parser.add_argument(
        "--check_perplexity",
        action="store_true",
        help="Evaluate perplexity of the original model",
    )

    parser.add_argument(
        "--spbits",
        type=int,
        default=8,
        help="Bit width for high precision weights",
    )

    parser.add_argument(
        "--block_mask_density",
        type=float,
        default=0.5,
        help="Threshold for deciding if a block should be marked as an outlier (sparse) block",
    )


    parser.add_argument(
        "--plot",
        action="store_true",
        help="To print all the plots on the pdf"
    )

    parser.add_argument(
        "--quant_error_title",
        type=str,
        default="quant_error.npy",
        help="Title for npy file storing all quant error weights"
    )

    args = parser.parse_args()

    if args.dataset == "custom":
        print(
            "WARNING: `--custom_data_path` argument and `--dataset=custom` option are DEPRECATED. ",
            "Pass dataset path directly to `dataset` argument or use 'pajama', 'refinedweb'",
            "See README.md for examples.",
        )
        args.dataset = args.custom_data_path

    if args.wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        args.exp_name = (
            os.environ.get("WANDB_NAME", "SpQR_run")
            + f"_wbits_{args.wbits}"
            + f"_groupsize_{args.groupsize}"
            + f"_qq_scale_bits_{args.qq_scale_bits}"
            + f"_qq_zero_bits_{args.qq_zero_bits}"
            + f"_qq_groupsize_{args.qq_groupsize}"
            + f"_outl_{args.outlier_threshold}"
            + f"_permord_{args.permutation_order}"
            + f"{'_new_eval' if args.new_eval else ''}"
        )
        wandb.init(
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
        )
        wandb.run.log_code(".")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache() 

    print("============ Initial model perplexities ============")
    if args.check_perplexity:
        config = AutoConfig.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, torch_dtype=torch.float16, device_map="auto")
        model.seqlen = 2048
        if model.config.model_type.lower() not in ["vit"]:
            datasets = ["wikitext2", "ptb", "c4"]
            # datasets = []
        if args.new_eval and model.config.model_type.lower() not in ["vit"]:
            datasets = ["wikitext2", "ptb-new", "c4-new"]
            # datasets = []
        for dataset in datasets:
            testloader = get_loaders(
                dataset,
                seed=args.seed,
                model_path=args.model_path,
                seqlen=model.seqlen,
                eval_mode=True,
                subsample = args.subsample
            )
            args.dataset_name = dataset
            perplexity_eval(model, testloader, args, device)

        del model
        torch.cuda.empty_cache()

    print("============  Loading model... ============")
    # model = get_model(args.model_path, args.sparse, args.load, args.dtype, args.density).train(False)
    model = get_quantized_block_mask_model(args.model_path, args.dataset, args.nsamples, args.seed, args, args.sparse, args.load, args.dtype, args.subsample).train(False)
    # model = torch.load("/home/lj9979/pytorch_block_sparse/pytorch_block_sparse/model.pt")

    print("\n============ Quantizing model... ============")
    if args.wbits < 16 and args.load:
        print("\n Warning: You are quantizing quantized model!")
    quantize_model(model, args, device)
    # quantize_model(model, args, "cpu")

    if model.config.model_type.lower() in ["vit"]:
        print("\n============ Testing the model... ============")
        normalize = transforms.Normalize(0.5, 0.5)
        val_transforms = transforms.Compose([
            transforms.Resize(384, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            normalize,
        ])
        imagenet_data = datasets.ImageNet('/home/lj9979/PyTorch-Pretrained-ViT/examples/imagenet/data/', split='val', transform = val_transforms)
        val_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        best_acc1 = 0
        for epoch in range(1):
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
    
    print("\n============ Evaluating perplexity... ============")
    torch.cuda.reset_peak_memory_stats()
    datasets = []
    if model.config.model_type.lower() not in ["vit"]:
        datasets = ["wikitext2", "ptb", "c4"]
        # datasets = []
    if args.new_eval and model.config.model_type.lower() not in ["vit"]:
        datasets = ["wikitext2", "ptb-new", "c4-new"]
        # datasets = []
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            seed=args.seed,
            model_path=args.model_path,
            seqlen=model.seqlen,
            eval_mode=True,
        )
        args.dataset_name = dataset
        
        perplexity_eval(model, testloader, args, device)

    print(f"eval: {torch.cuda.max_memory_allocated()=:,}")
    if args.wandb:
        wandb.log({"max_cuda_mem_eval": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
