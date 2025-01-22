from contextlib import contextmanager
import time
import os

import torch
import torch.nn as nn
from tqdm import trange
from transformers import AutoConfig, AutoModelForCausalLM, BertModel, AutoTokenizer
from pytorch_block_sparse import BlockSparseLinear, BlockSparseModelPatcher
# from custom_mm import BlockedEllModelPatcher
from pytorch_pretrained_vit import ViT
import re
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from quant_groups import dequantize
from datautils import get_loaders
from spqr_engine import Quantizer, SPQRUtil, quantize

MODEL_ERROR_MSG = "Unsupported model type {} - only 'llama', 'Yi', 'opt' and 'falcon' are supported"
FALCON_TYPES = ("falcon", "refinedweb", "refinedwebmodel")
LLAMA_LIKE = ("llama", "Yi")

@torch.no_grad()
def get_inps(model, data_iterable, args, dev, nsamples=None):
    """mocks model launch to collect inputs to the first model layer"""
    print("catching inputs from data", flush=True)

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
            "attention_mask",
        ]
    if model.config.model_type.lower() in FALCON_TYPES:
        forward_arg_names.append("alibi")

    cache = {"i": 0, "attention_mask": None, "mask":None, "alibi": None}

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
def quantize_spqr_full_model(model, dataloader, args, device):
    print("\nStarting SPQR quantization with Block Sparsity ...")

    inps, forward_args = get_inps(model, dataloader, args, dev="cpu" if args.offload_activations else device)
    outs = torch.zeros_like(inps)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    save = getattr(args, "save", False)

    total_model_bits = 0
    normal_outlier_count_global, w_count_global = 0, 0

    layers = get_layers(model)

    mp = BlockSparseModelPatcher()

    reg_patchables, patchables = mp.get_patchable_layers(model)

    if model.config.model_type == "opt" and "opt-350m" in args.model_path:
        patchables = patchables[2:]

    del reg_patchables

    # pdf = PdfPages("mask_pattern.pdf")

    quant_sensitivities_dict = {}
    
    # for i in [30]:
    for i in range(len(layers)):
        print(f"\n---------------- Layer {i} of {len(layers)} ----------------")
        normal_outlier_count, w_count = 0, 0
        stats_payload = {}

        layer_dev_original = next(layers[i].parameters()).device  # quantized layer will return there
        print(f"{layer_dev_original=}")
        if layer_dev_original.type != "cuda":
            layer = layers[i].to(device)
        else:
            layer = layers[i]
        layer_dev = next(layers[i].parameters()).device
        all_sublayers = find_sublayers(layer)

        for k, v in forward_args.items():
            forward_args[k] = v.to(layer_dev) if isinstance(v, torch.Tensor) else v

        if args.true_sequential:
            sequential = get_sequential_groups(model)
        else:
            sequential = [list(all_sublayers.keys())]

        for names in sequential:
            subset = {n: all_sublayers[n] for n in names}

            spqr_handlers = {}
            for sublayer_name in subset:
                spqr_handlers[sublayer_name] = SPQRUtil(subset[sublayer_name])

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

            count = 0
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
                    block_shape=(args.block_shape_width, args.block_shape_height),
                    sparse = False,
                    plot = args.plot,
                    fig_title  = sublayer_name+"_"+str(i),
                    layer_no = i,
                    block_mask_density = args.block_mask_density,
                    pdf = pdf,
                )

                # spqr_handlers[sublayer_name].layer.weight.data = quantized.weight.to(
                #     spqr_handlers[sublayer_name].layer.weight.data.dtype
                # )

                # OUTLIER STATS per module:
                normal_outliers_count = quantized.unstructured_outlier_mask.to(torch.int32).sum()
                stats_payload[f"n_{sublayer_name}_ol_share"] = (normal_outliers_count / quantized.weight.numel()).item()
                normal_outlier_count += normal_outliers_count.item()
                w_count += quantized.weight.numel()
                block_mask = quantized.block_mask
                perc_out = quantized.perc_out
                quant_sensitivities_dict[sublayer_name+"_"+str(i)] = quantized.quantization_errors.cpu()

                # total_model_bits += num_bits
                print("Percentage of outliers: ", perc_out)

                # Add the pattern for the block mask
                print(f"Adding pattern for {patchables[i*len(subset) + count]} of layer {i}")
                mp.add_pattern(patchables[i*len(subset) + count], {"density":0.1}, (args.block_shape_width, args.block_shape_height), block_mask)
                count += 1

        layers[i] = layer.to(layer_dev_original)
        del layer
        del spqr_handlers
        torch.cuda.empty_cache()

        normal_outlier_count_global += normal_outlier_count
        w_count_global += w_count

    np.save(args.quant_error_title, quant_sensitivities_dict)
    pdf.close()
    del layers
    del inps
    del forward_args
    torch.cuda.empty_cache()

    mp.patch_model(model)
    print("Total number of bits used for quantization: ", total_model_bits)
    print("Model patched sucessfully ...")


    # added to perform compuatation on cpu, remove later for cuda
    # model.to('cpu')
    torch.save(model.state_dict(), "/home/lj9979/pytorch_block_sparse/pytorch_block_sparse/model.pt")

    print("=====================\nFinal stats:")
    print(f"global_ol_share:  {normal_outlier_count_global / w_count_global:.3%}")


    model.config.use_cache = use_cache
    print(f"quantize: {torch.cuda.max_memory_allocated()=:,}")
    return model


@contextmanager
def suspend_nn_inits():
    def skip(*args, **kwargs):
        pass

    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_  # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip  # replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring


def get_model(model_path, sparse=False, load_quantized=None, dtype="auto", density=0.1):
    if dtype == "auto":
        dtype = (
            AutoConfig.from_pretrained(model_path, trust_remote_code=True).torch_dtype or "auto"
        )  # force transformers 4.29.2 to follow the same rules as 4.30.x
    else:
        dtype = getattr(torch, dtype)

    with suspend_nn_inits():
        if load_quantized:
            print("Initializing model with random weights...")
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)  # consider trust_remote_code=True
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=dtype).eval()
            print("Loading quantized model ...")
            model = load_quantized_model(model, load_quantized)
        elif sparse:
            if "vit" in model_path:
                model_name = 'B_16_imagenet1k'
                model = ViT(model_name, pretrained=True)
            elif "bert" in model_path:
                model = BertModel.from_pretrained("bert-base-uncased")
            else:
                config = AutoConfig.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16)
            # mp = BlockedEllModelPatcher()
            mp = BlockSparseModelPatcher()

            patchables = mp.get_patchable_layers(model)

            dedup_layers = []

            for patchable in patchables:
                r = patchable["regexp"]
                r = re.sub(r'[0-9]+', '[0-9]+', r)
                if r not in dedup_layers and r not in ["fc", "lm_head"]:
                    dedup_layers.append(r)
            
            for i in range(len(dedup_layers)):
                
                mp.add_pattern(dedup_layers[i], {"density":density})

            # model.to('cuda:0')

            mp.patch_model(model)
            print("Model patched sucessfully ...")

        elif "vit" in model_path:
            model_name = 'B_16_imagenet1k'
            model = ViT(model_name, pretrained=True).to('cuda:0')

        else:
            print("Loading pretrained model ...")
            config = AutoConfig.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=dtype,)
    model.seqlen = 2048

    print("Model loaded sucessfully ...")

    return model

def get_quantized_block_mask_model(model_path, dataset, nsamples, seed, args, sparse=False, load_quantized=None, dtype="auto", subsample = -1):
    if dtype == "auto":
        dtype = (
            AutoConfig.from_pretrained(model_path, trust_remote_code=True).torch_dtype or "auto"
        )  # force transformers 4.29.2 to follow the same rules as 4.30.x
    else:
        dtype = getattr(torch, dtype)

    with suspend_nn_inits():
        if load_quantized:
            print("Initializing model with random weights...")
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)  # consider trust_remote_code=True
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=dtype).eval()
            print("Loading quantized model ...")
            model = load_quantized_model(model, load_quantized)
        elif sparse:
            if "vit" in model_path:
                model_name = 'B_16_imagenet1k'
                model = ViT(model_name, pretrained=True)
            elif "bert" in model_path:
                model = BertModel.from_pretrained("bert-base-uncased")
            else:
                config = AutoConfig.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, device_map="auto")
            model.seqlen = 2048
            dataloader = get_loaders(
                            dataset,
                            nsamples=nsamples,
                            seed=seed,
                            model_path=model_path,
                            seqlen=model.seqlen,
                            subsample=subsample,
                        )
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = quantize_spqr_full_model(model, dataloader, args, device)
            

        elif "vit" in model_path:
            model_name = 'B_16_imagenet1k'
            model = ViT(model_name, pretrained=True).to('cuda:0')

        else:
            print("Loading pretrained model ...")
            config = AutoConfig.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=dtype,)
    model.seqlen = 2048

    print("Model loaded sucessfully ...")

    return model


def get_model_head(model):
    head = torch.nn.ModuleList()
    if model.config.model_type in LLAMA_LIKE:
        if model.model.norm is not None:
            head.append(model.model.norm)
        head.append(model.lm_head)
    elif model.config.model_type.lower() in FALCON_TYPES:
        if model.transformer.ln_f is not None:
            head.append(model.transformer.ln_f)
        head.append(model.lm_head)
    elif model.config.model_type == "opt":
        if model.model.decoder.final_layer_norm is not None:
            head.append(model.model.decoder.final_layer_norm)
        if model.model.decoder.project_out is not None:
            head.append(model.model.decoder.project_out)
        head.append(model.lm_head)
    elif model.config.model_type == "vit":
        head.append(model.norm)
        head.append(model.fc)
    elif model.config.model_type == "bert":
        head.append(model.pooler)
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))
    return head


def get_lm_logits(inps_, model):
    if model.config.model_type in LLAMA_LIKE:
        hidden_states = inps_.unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    elif model.config.model_type.lower() in FALCON_TYPES:
        hidden_states = inps_.unsqueeze(0)
        if model.transformer.ln_f is not None:
            hidden_states = model.transformer.ln_f(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    elif model.config.model_type == "opt":
        hidden_states = inps_.unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    elif model.config.model_type == "vit":
        hidden_states = inps_.unsqueeze(0)
        lm_logits = model.fc(hidden_states)
    elif model.config.model_type == "bert":
        hidden_states = inps_.unsqueeze(0)
        lm_logits = model.pooler(hidden_states)
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))
    return lm_logits


def get_layers(model):
    if model.config.model_type in LLAMA_LIKE:
        return model.model.layers
    elif model.config.model_type.lower() in FALCON_TYPES:
        return model.transformer.h
    elif model.config.model_type == "opt":
        return model.model.decoder.layers
    elif model.config.model_type == "vit":
        return model.transformer.blocks
    elif model.config.model_type == "bert":
        return model.encoder.layer
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))


def find_sublayers(module, layers=(nn.Conv2d, nn.Linear, BlockSparseLinear), check=False):
    res = {}
    for name, layer in module.named_modules():
        # if check:
        #     if isinstance(layer, (BlockSparseLinear)):
        #         res[name] = layer
        # else:
        #     if isinstance(layer, layers):
        #         res[name] = layer
        if isinstance(layer, layers):
            res[name] = layer

    return res


def get_sequential_groups(model):
    if model.config.model_type in LLAMA_LIKE:
        return [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]
    elif model.config.model_type.lower() in FALCON_TYPES:
        return [
            ["self_attention.query_key_value"],
            ["self_attention.dense"],
            ["mlp.dense_h_to_4h"],
            ["mlp.dense_4h_to_h"],
        ]
    elif model.config.model_type == "opt":
        return [
            ["self_attn.q_proj"],
            ["self_attn.k_proj"],
            ["self_attn.v_proj"],
            ["self_attn.out_proj"],
            ["fc1"],
            ["fc2"],
        ]
    elif model.config.model_type == "vit":
        return [
            ["attn.proj_q"],
            ["attn.proj_k"],
            ["attn.proj_v"],
            ["proj"],
            ["pwff.fc1"],
            ["pwff.fc2"],
        ]
    elif model.config.model_type == "bert":
        return [
            ["attention.self.query"],
            ["attention.self.key"],
            ["attention.self.value"],
            ["attention.output.dense"],
            ["intermediate.dense"],
            ["output.dense"],
        ]
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))


def read_quant_weight_from_file(load_path, block_i, layer_name):
    return torch.load(load_path + "/" + str(block_i) + "/" + layer_name)


def load_quantized_model(model, load_path):
    layers = get_layers(model)
    for i in trange(len(layers)):
        layer = layers[i]
        sub_layers = find_sublayers(layer)
        for name in sub_layers:
            quantized_params_dict = read_quant_weight_from_file(load_path, i, name)
            sub_layers[name].weight = nn.Parameter(
                layer_weight_dequantization(quantized_params_dict).to(sub_layers[name].weight.data.dtype)
            )
        layers[i] = layer
    model.load_state_dict(torch.load(load_path + "/not_quantized_weights.pt"), strict=False)
    return model


def layer_weight_dequantization(quantized_params_dict):
    out_dim, in_dim = quantized_params_dict["weight_shape"]
    blocksize = quantized_params_dict["blocksize"]
    keep_last_columns = quantized_params_dict["keep_last_columns"]
    reconstructed_weight = torch.zeros(quantized_params_dict["weight_shape"])
    block_start_iter = range(0, in_dim - keep_last_columns, blocksize)
    block_start_iter = block_start_iter
    current_ind = 0

    for block_start in block_start_iter:
        block_end = min(block_start + blocksize, in_dim)
        for column_index in range(block_start, block_end):
            if column_index % quantized_params_dict["groupsize"] == 0:
                if quantized_params_dict["quant_layer_scale_qq_scale"]:
                    dequantize_zeros = dequantize(
                        quantized_params_dict["quant_layer_zeros"][current_ind],
                        quantized_params_dict["quant_layer_zero_qq_scale"][current_ind],
                        quantized_params_dict["quant_layer_zero_qq_zero"][current_ind],
                    )
                    dequantize_scale = dequantize(
                        quantized_params_dict["quant_layer_scale"][current_ind],
                        quantized_params_dict["quant_layer_scale_qq_scale"][current_ind],
                        quantized_params_dict["quant_layer_scale_qq_zero"][current_ind],
                    )
                else:
                    dequantize_zeros = quantized_params_dict["quant_layer_zeros"][current_ind]
                    dequantize_scale = quantized_params_dict["quant_layer_scale"][current_ind]
                current_ind += 1

            reconstructed_weight[:, column_index] = dequantize(
                quantized_params_dict["quant_weights"][:, column_index].unsqueeze(1),
                dequantize_scale.reshape(-1, 1),
                dequantize_zeros.reshape(-1, 1),
            ).reshape_as(reconstructed_weight[:, column_index])
    reconstructed_weight = (
        reconstructed_weight * (quantized_params_dict["outliers_matrix"].to_dense().cpu() == 0)
        + quantized_params_dict["outliers_matrix"].to_dense().cpu()
    )
    invperm = torch.argsort(quantized_params_dict["perm"]).cpu()
    reconstructed_weight = reconstructed_weight[:, invperm]
    return reconstructed_weight
