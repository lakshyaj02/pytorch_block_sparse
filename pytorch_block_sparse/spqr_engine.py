from __future__ import annotations

import os
import sys
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import math
from typing import NamedTuple, Optional, Union

import torch
from tqdm.auto import tqdm

from quant_groups import Quantizer, dequantize, quantize
from weight_permutation import get_permutation_order

class WeightOnlyQuantConfig:
    def __init__(self, algorithm):
        """This is the Base class for Weight Only Quant Configuration.

        Args:
            algorithm:
                weight only quantize algorithm name.
        """
        self.algorithm = algorithm

class RTNWeightOnlyQuantConfig(WeightOnlyQuantConfig):
    def __init__(
        self,
        ratios=None,
    ):
        """
        This is a class for round-to-nearest (RTN) algorithm Weight Only Quant Configuration.
        RTN is the most straightforward way to quantize weight using scale maps.

        Args:
            ratios:
                percentile of clip. Defaults to {}.
        """
        if ratios is None:
            ratios = {}
        super().__init__(
            algorithm="RTN",
        )
        self.ratios = ratios

class SPQRUtil:
    """Learns GPTQ for a single linear layer"""

    def __init__(self, layer, sparse=False):
        self.layer = layer
        self.sparse = sparse
        if sparse:
            self.sparse_dev = self.layer.sparse_weight.data.device
            self.sparse_columns = self.layer.sparse_weight.sparse_data.shape[1]
            self.dense_dev = layer.sparse_weight.dense_data.device
            self.dense_columns = self.layer.sparse_weight.dense_data.shape[1]

        else:
            self.dev = layer.weight.device
            self.columns = self.layer.weight.data.shape[1]
            self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        
        self.nsamples = 0
        if sparse:
            self.sparse_H = torch.zeros((self.sparse_columns, self.sparse_columns), device=self.sparse_dev)
            self.dense_H = torch.zeros((self.dense_columns, self.dense_columns), device=self.dense_dev)

    def add_batch(self, inp):
        if self.sparse: 
            assert self.sparse_H is not None, "Already ran quantization; cannot add more data batches"
            assert self.dense_H is not None, "Already ran quantization; cannot add more data batches"
        else:
            assert self.H is not None, "Already ran quantization; cannot add more data batches"
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        if self.sparse:
            self.sparse_H *= self.nsamples / (self.nsamples + tmp)
            self.dense_H *= self.nsamples / (self.nsamples + tmp)
        else:
            self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        
        if self.sparse:
            self.sparse_H += inp.matmul(inp.t())
            self.dense_H += inp.matmul(inp.t())
        else: 
            self.H += inp.matmul(inp.t())

    def save_quantization(self, quantizer, save_quant_dict):
        if quantizer.qq_scale_bits is not None:
            save_quant_dict["quant_layer_scale"].append(quantizer.quant_scale.to(torch.int8))
            save_quant_dict["quant_layer_scale_qq_scale"].append(
                quantizer.qq_scale.scale.to(save_quant_dict["save_float_dtype"])
            )
            save_quant_dict["quant_layer_scale_qq_zero"].append(
                quantizer.qq_scale.zero.to(save_quant_dict["save_float_dtype"])
            )
        else:
            save_quant_dict["quant_layer_scale"].append(
                quantizer.scale.to(save_quant_dict["save_float_dtype"])
            )

        if quantizer.qq_zero_bits is not None and (
            (not quantizer.round_zero) or quantizer.qq_zero_bits < quantizer.bits
        ):
            save_quant_dict["quant_layer_zeros"].append(quantizer.quant_zero.to(torch.int8))
            save_quant_dict["quant_layer_zero_qq_scale"].append(
                quantizer.qq_zero.scale.to(save_quant_dict["save_float_dtype"])
            )
            save_quant_dict["quant_layer_zero_qq_zero"].append(
                quantizer.qq_zero.zero.to(save_quant_dict["save_float_dtype"])
            )
        else:
            save_quant_dict["quant_layer_zeros"].append(
                quantizer.zero.to(save_quant_dict["save_float_dtype"])
            )

        return quantizer, save_quant_dict

    
    def plot_mask(self, pdf, mask, fig_title, outier_sensitivities=None):
        """
        Function to visualize the weight mask and sensitivities
        """
        mask = mask.cpu()
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(mask, cmap='hot', interpolation='nearest')
        plt.title(fig_title)
        pdf.savefig(fig)  # saves the current figure into a pdf page
        plt.close()

    def quantize(
        self,
        *,
        bits: int = 2,
        blocksize: int = 128,
        percdamp: float = 1e-2,
        groupsize: Optional[int] = None,
        keep_last_columns: int = 0,
        outlier_relative_threshold: float = float("inf"),
        permutation_order: Union[str, torch.Tensor] = "identity",
        keep_H: bool = True,
        simplified_outliers: bool = False,
        verbose=True,
        perchannel: bool = True,
        sym: bool = False,
        save_quantization: bool = False,
        sparse: bool = False,
        block_shape: tuple = (32,32),
        get_layer_size: bool = True,
        spbits: int = 8,
        plot: bool = False,
        fig_title: str = "weight_layer",
        layer_no: int = -1,
        pdf = None,
        block_mask_density: float = 0.5,
        **kwargs,
    ) -> QuantizationResult:
        """
        :param bits: number of bits used at the lowest level (the full model size will be different!)
        :param blocksize: take blocks of this many input features at a time for GPTQ
        :note: blocksize affects runtime and memory, but does not affect the resulting matrix (up to machine precision)
        :param groupsize: fit quantization scaling / statistics to each group of this many input features
        :param percdamp: relative regularizer added to hessian diagonal before inversion
        :note: if groupsize_in_dim* is None, use the same quantization statistics across all input features
        :param keep_last_columns: if not None, keep the last (this many) input features un_quantized and return them
        :note: the un-quantized columns will be a part of the first returned result
        :param outlier_relative_threshold: threshold used for *UNSTRUCTURED* outliers, relative to
        :note: if keep_last_columns > 0, quantized_dequantized_weights[-keep_last_columns:] will be non-quantized
        :param permutation_order: re-order input features using a certain policy
        :param keep_H: if False, delete the accumulated hessian during quantize; if False, keep the accumulated hessian
        :param simplified_outliers: if True,do not perform leave-one-out evaluation when detecting outliers;
            works faster, but generally worse in perplexity
        :param verbose: if True, display a tqdm progressbar over input columns
        :param sym: if True, base weight quantization is symmetric
        :param perchannel: if True, base weight quantization will learn statistics for each output dimension separately
        :return: a QuantizationResult tuple that contains(
            weight, perm, _unused, _unused, _unused, _unused, quantization_errors, outlier_unstructured_mask
        ), see class QuantizationResult below for details
        """
        if sparse:
            sparse_weight = self.layer.sparse_weight.sparse_data.detach().to(dtype=torch.float, copy=True)
            dense_weight = self.layer.sparse_weight.dense_data.detach().to(dtype=torch.float, copy=True)
            sparse_save_quant_dict = {}
            dense_save_quant_dict = {}
        else:
            weight = self.layer.weight.detach().to(dtype=torch.float, copy=True)

        save_quant_dict = {}
        
        if sparse:
            sparse_perm = get_permutation_order(self.sparse_H, sparse_weight, permutation_order)
            dense_perm = get_permutation_order(self.dense_H, dense_weight, permutation_order)
        else:
            perm = get_permutation_order(self.H, weight, permutation_order)

        if save_quantization:
            save_quant_dict["quant_weights"] = []
            save_quant_dict["quant_layer_scale"] = []
            save_quant_dict["quant_layer_zeros"] = []
            save_quant_dict["quant_layer_scale_qq_scale"] = []
            save_quant_dict["quant_layer_scale_qq_zero"] = []
            save_quant_dict["quant_layer_zero_qq_scale"] = []
            save_quant_dict["quant_layer_zero_qq_zero"] = []
            save_quant_dict["save_float_dtype"] = self.layer.weight.dtype
            save_quant_dict["outliers_matrix"] = torch.zeros(
                weight.shape, dtype=save_quant_dict["save_float_dtype"]
            ).to(
                weight.device
            )  # shape = [out_features, in_features]

        if sparse:
            sparse_weight = sparse_weight[:, sparse_perm]
            dense_weight = dense_weight[:, dense_perm]
        else:
            weight = weight[:, perm]  # note: weight is modified
        
        if sparse:
            sparse_H = self.sparse_H
            dense_H = self.dense_H
        else:
            H = self.H
        if keep_H:
            if sparse:
                sparse_H = sparse_H.clone()
                dense_H = dense_H.clone()
            else:
                H = H.clone()  # protect from in-place changes
        else:
            if sparse:
                sparse_H = None
                dense_H = None
            else:
                self.H = None

        if sparse:
            sparse_H = sparse_H[sparse_perm][:, sparse_perm]
            dense_H = dense_H[dense_perm][:, dense_perm]
        else:
            H = H[perm][:, perm]

        if sparse:
            self.sparse_dead = torch.diag(sparse_H) == 0
            self.dense_dead = torch.diag(dense_H) == 0
        else:
            self.dead = torch.diag(H) == 0  # indices of input features that do not affect outputs
        
        if percdamp > 0:
            if sparse:
                sparse_ix = torch.arange(len(sparse_H), device=sparse_weight.device)
                dense_ix = torch.arange(len(dense_H), device=dense_weight.device)
                sparse_H[sparse_ix, sparse_ix] += percdamp * abs(torch.diag(sparse_H)).mean()
                dense_H[dense_ix, dense_ix] += percdamp * abs(torch.diag(dense_H)).mean()
                del sparse_ix, dense_ix
            else:
                ix = torch.arange(len(H), device=weight.device)
                H[ix, ix] += percdamp * abs(torch.diag(H)).mean()
                del ix
            

        if sparse:
            sparse_H[self.sparse_dead, self.sparse_dead] = 1
            sparse_weight[:, self.sparse_dead] = 0
            dense_H[self.dense_dead, self.dense_dead] = 1
            dense_weight[:, self.dense_dead] = 0
            sparse_H_inv = torch.cholesky_inverse(torch.linalg.cholesky(sparse_H))
            dense_H_inv = torch.cholesky_inverse(torch.linalg.cholesky(dense_H))
            sparse_H_inv_cho = torch.linalg.cholesky(sparse_H_inv, upper=True)
            dense_H_inv_cho = torch.linalg.cholesky(dense_H_inv, upper=True)
            sparse_H_inv_cho_diag = torch.diag(sparse_H_inv_cho)
            dense_H_inv_cho_diag = torch.diag(dense_H_inv_cho)
            del sparse_H, dense_H
        else:
            H[self.dead, self.dead] = 1
            weight[:, self.dead] = 0
            H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            H_inv_cho = torch.linalg.cholesky(H_inv, upper=True)
            H_inv_cho_diag = torch.diag(H_inv_cho)
            del H

        if sparse:
            sparse_bits = spbits
            sparse_quantizer = Quantizer()
            dense_quantizer = Quantizer()
            sparse_quantizer.configure(sparse_bits, perchannel=perchannel, sym=sym, **kwargs)
            dense_quantizer.configure(bits, perchannel=perchannel, sym=sym, **kwargs)
        else:
            sparse_bits = spbits
            quantizer = Quantizer()
            quantizer.configure(bits, perchannel=perchannel, sym=sym, **kwargs)
        
        if sparse:
            assert sparse_H_inv_cho.shape[0] == sparse_H_inv_cho.shape[1] == sparse_weight.shape[1], "weight must be [out_features, in_features]"
            assert dense_H_inv_cho.shape[0] == dense_H_inv_cho.shape[1] == dense_weight.shape[1], "weight must be [out_features, in_features]"
        else:
            assert H_inv_cho.shape[0] == H_inv_cho.shape[1] == weight.shape[1], "weight must be [out_features, in_features]"
        if sparse:
            sparse_out_dim, sparse_in_dim = sparse_weight.shape  # [out_features, in_features]
            dense_out_dim, dense_in_dim = dense_weight.shape  # [out_features, in_features]
        else:
            out_dim, in_dim = weight.shape  # [out_features, in_features]

        if groupsize is None:
            if sparse:
                groupsize = dense_in_dim
            else:
                groupsize = in_dim

        # prepare outlier detection
        if sparse:
            # sparse_outlier_column_indices = torch.empty(0, dtype=torch.int64, device=sparse_weight.device)
            # dense_outlier_column_indices = torch.empty(0, dtype=torch.int64, device=dense_weight.device)
            del sparse_H_inv
            del dense_H_inv
        else:
            # outlier_column_indices = torch.empty(0, dtype=torch.int64, device=weight.device)
            del H_inv


        if sparse:
            sparse_outlier_scale = (sparse_weight.var(dim=0) / torch.diag(sparse_H_inv_cho).square()).mean().item()
            dense_outlier_scale = (dense_weight.var(dim=0) / torch.diag(dense_H_inv_cho).square()).mean().item()
            # sparse_unstructured_outlier_threshold = outlier_relative_threshold * sparse_outlier_scale
            sparse_unstructured_outlier_threshold = float("inf")* sparse_outlier_scale
            # dense_unstructured_outlier_threshold = outlier_relative_threshold * dense_outlier_scale
            dense_unstructured_outlier_threshold = float("inf")* dense_outlier_scale
            in_group_index = -1
        else:
            outlier_scale = (weight.var(dim=0) / torch.diag(H_inv_cho).square()).mean().item()
            unstructured_outlier_threshold = outlier_relative_threshold * outlier_scale
            in_group_index = -1  # index of current group of input features, for group quantizer purposes

        if sparse:
            sparse_quantization_errors = torch.zeros_like(sparse_weight)
            sparse_unstructured_outlier_mask = torch.zeros_like(sparse_weight, dtype=torch.bool)
            dense_quantization_errors = torch.zeros_like(dense_weight)
            dense_unstructured_outlier_mask = torch.zeros_like(dense_weight, dtype=torch.bool)
            sparse_block_start_iter = range(0, sparse_in_dim - keep_last_columns, blocksize)
            dense_block_start_iter = range(0, dense_in_dim - keep_last_columns, blocksize)
            sparse_block_start_iter = tqdm(sparse_block_start_iter, leave=False) if verbose else sparse_block_start_iter
            dense_block_start_iter = tqdm(dense_block_start_iter, leave=False) if verbose else dense_block_start_iter
        else:
            quantization_errors = torch.zeros_like(weight)
            unstructured_outlier_mask = torch.zeros_like(weight, dtype=torch.bool)
            block_start_iter = range(0, in_dim - keep_last_columns, blocksize)
            block_start_iter = tqdm(block_start_iter, leave=False) if verbose else block_start_iter

        if sparse:
            for block_start in sparse_block_start_iter:
                block_end = min(block_start + blocksize, sparse_in_dim)
                for column_index in range(block_start, block_end):
                    if column_index % groupsize == 0:
                        # fit weight quantizer on the upcoming group of weight columns (inputs), across all rows (outputs)
                        in_group_index += 1
                        group_weight = sparse_weight[:, column_index : column_index + groupsize]

                        if simplified_outliers or (sparse_unstructured_outlier_threshold == float("inf")):
                            sparse_quantizer.find_params(group_weight, weight=True)

                        else:
                            # objective: detect which weights will be designated as outliers, fit quantizer *without* these weights
                            assert perchannel, "refitting quantizer is only implemented for perchannel=True"
                            group_diag_sparse_hessian_inv_cho = sparse_H_inv_cho_diag[column_index : column_index + groupsize]
                            loo_quantization_error_sq = get_leave_one_out_error(
                                group_weight, group_diag_sparse_hessian_inv_cho, bits=sparse_bits, sym=sym
                            )
                            # ^-- dequantized(quantized(group_weight)) using a quantizer trained on all weights except the reconstructed one

                            likely_unstructured_outlier_mask = (
                                loo_quantization_error_sq > sparse_unstructured_outlier_threshold
                            ).float()

                            non_outlier_mask = 1 - likely_unstructured_outlier_mask
                            mean_over_non_outliers = torch.sum(
                                group_weight * non_outlier_mask, dim=1, keepdim=True
                            ) / torch.sum(non_outlier_mask, dim=1, keepdim=True).clamp_min(1)
                            group_weight_without_outliers = group_weight * non_outlier_mask + mean_over_non_outliers * (
                                1 - non_outlier_mask
                            )
                            sparse_quantizer.find_params(group_weight_without_outliers, weight=True)
                            del group_diag_sparse_hessian_inv_cho, loo_quantization_error_sq
                            del mean_over_non_outliers, group_weight_without_outliers, non_outlier_mask

                        if save_quantization:
                            sparse_quantizer, sparse_save_quant_dict = self.save_quantization(sparse_quantizer, sparse_save_quant_dict)
                        del group_weight

                    sparse_weight_quant_i = quantize(
                        sparse_weight[:, column_index].unsqueeze(1), sparse_quantizer.scale, sparse_quantizer.zero, sparse_quantizer.maxq
                    )
                    sparse_weight_i_quantized = dequantize(sparse_weight_quant_i, sparse_quantizer.scale, sparse_quantizer.zero).reshape_as(
                        sparse_weight[:, column_index]
                    )

                    delta_weight_i = sparse_weight[:, column_index] - sparse_weight_i_quantized  # [out_dim]
                    sparse_quantization_errors[:, column_index] = (
                        delta_weight_i / sparse_H_inv_cho[column_index, column_index]
                    )  # [out_dim]
                    
                    if sparse_unstructured_outlier_threshold != float("inf"):
                        sparse_unstructured_outlier_mask[:, column_index] = (
                            sparse_quantization_errors[:, column_index].square() > sparse_unstructured_outlier_threshold
                        )
                        # re-quantize without outliers
                        is_outlier = sparse_unstructured_outlier_mask[:, column_index].float()

                        sparse_weight_quant_i = quantize(
                            (sparse_weight[:, column_index] * (1 - is_outlier)).unsqueeze(1),
                            sparse_quantizer.scale,
                            sparse_quantizer.zero,
                            sparse_quantizer.maxq,
                        )
                        weight_i_quantized_wo_outliers = dequantize(
                            sparse_weight_quant_i, sparse_quantizer.scale, sparse_quantizer.zero
                        ).reshape_as(sparse_weight[:, column_index])
                        sparse_weight_i_quantized = (
                            weight_i_quantized_wo_outliers * (1 - is_outlier) + sparse_weight[:, column_index] * is_outlier
                        )  # [out_dim]

                        if save_quantization:
                            sparse_save_quant_dict["outliers_matrix"][:, column_index] = sparse_weight[:, column_index] * is_outlier

                        del weight_i_quantized_wo_outliers

                        delta_weight_i = sparse_weight[:, column_index] - sparse_weight_i_quantized  # [out_dim]
                        sparse_quantization_errors[:, column_index] = (
                            delta_weight_i / sparse_H_inv_cho[column_index, column_index]
                        )  # [out_dim]

                    if save_quantization:
                        sparse_save_quant_dict["quant_weights"].append(sparse_weight_quant_i.to(torch.int8))

                    sparse_weight[:, column_index] = sparse_weight_i_quantized
                    sparse_weight[:, column_index + 1 : block_end].addr_(
                        sparse_quantization_errors[:, column_index],
                        sparse_H_inv_cho[column_index, column_index + 1 : block_end],
                        alpha=-1,
                    )

                sparse_weight[:, block_end:].addmm_(
                    sparse_quantization_errors[:, block_start:block_end],
                    sparse_H_inv_cho[block_start:block_end, block_end:],
                    alpha=-1,
                )

            if permutation_order != "identity":
                invperm = torch.argsort(sparse_perm)
                sparse_weight = sparse_weight[:, invperm]

            if save_quantization:
                sparse_save_quant_dict["perm"] = sparse_perm.to(torch.int32)
                sparse_save_quant_dict["keep_last_columns"] = 0
                sparse_save_quant_dict["blocksize"] = 128
                sparse_save_quant_dict["weight_shape"] = sparse_weight.shape
                sparse_save_quant_dict["groupsize"] = groupsize if groupsize else sparse_weight.shape[1]
                sparse_save_quant_dict["quant_weights"] = torch.cat(sparse_save_quant_dict["quant_weights"], dim=1)
                sparse_save_quant_dict["outliers_matrix"] = sparse_save_quant_dict["outliers_matrix"].to_sparse()

            # Dense counterpart
            for block_start in dense_block_start_iter:
                block_end = min(block_start + blocksize, dense_in_dim)
                for column_index in range(block_start, block_end):
                    if column_index % groupsize == 0:
                        # fit weight quantizer on the upcoming group of weight columns (inputs), across all rows (outputs)
                        in_group_index += 1
                        group_weight = dense_weight[:, column_index : column_index + groupsize]

                        if simplified_outliers or (dense_unstructured_outlier_threshold == float("inf")):
                            dense_quantizer.find_params(group_weight, weight=True)

                        # if simplified_outliers or (dense_unstructured_outlier_threshold == float("inf")):
                        else:
                            # objective: detect which weights will be designated as outliers, fit quantizer *without* these weights
                            assert perchannel, "refitting quantizer is only implemented for perchannel=True"
                            group_diag_sparse_hessian_inv_cho = dense_H_inv_cho_diag[column_index : column_index + groupsize]
                            loo_quantization_error_sq = get_leave_one_out_error(
                                group_weight, group_diag_sparse_hessian_inv_cho, bits=bits, sym=sym
                            )
                            # ^-- dequantized(quantized(group_weight)) using a quantizer trained on all weights except the reconstructed one

                            likely_unstructured_outlier_mask = (
                                # loo_quantization_error_sq > dense_unstructured_outlier_threshold
                                loo_quantization_error_sq > 0.1
                            ).float()

                            non_outlier_mask = 1 - likely_unstructured_outlier_mask
                            mean_over_non_outliers = torch.sum(
                                group_weight * non_outlier_mask, dim=1, keepdim=True
                            ) / torch.sum(non_outlier_mask, dim=1, keepdim=True).clamp_min(1)
                            group_weight_without_outliers = group_weight * non_outlier_mask + mean_over_non_outliers * (
                                1 - non_outlier_mask
                            )
                            dense_quantizer.find_params(group_weight_without_outliers, weight=True)
                            del group_diag_sparse_hessian_inv_cho, loo_quantization_error_sq
                            del mean_over_non_outliers, group_weight_without_outliers, non_outlier_mask

                        if save_quantization:
                            dense_quantizer, dense_save_quant_dict = self.save_quantization(dense_quantizer, dense_save_quant_dict)
                        del group_weight

                    dense_weight_quant_i = quantize(
                        dense_weight[:, column_index].unsqueeze(1), dense_quantizer.scale, dense_quantizer.zero, dense_quantizer.maxq
                    )
                    dense_weight_i_quantized = dequantize(dense_weight_quant_i, dense_quantizer.scale, dense_quantizer.zero).reshape_as(
                        dense_weight[:, column_index]
                    )

                    delta_weight_i = dense_weight[:, column_index] - dense_weight_i_quantized  # [out_dim]
                    dense_quantization_errors[:, column_index] = (
                        delta_weight_i / dense_H_inv_cho[column_index, column_index]
                    )  # [out_dim]
                    
                    if dense_unstructured_outlier_threshold != float("inf"):
                    # if simplified_outliers or (dense_unstructured_outlier_threshold == float("inf")):
                        dense_unstructured_outlier_mask[:, column_index] = (
                            # dense_quantization_errors[:, column_index].square() > dense_unstructured_outlier_threshold
                            dense_quantization_errors[:, column_index].square() > 0.1
                        )
                        # re-quantize without outliers
                        is_outlier = dense_unstructured_outlier_mask[:, column_index].float()

                        dense_weight_quant_i = quantize(
                            (dense_weight[:, column_index] * (1 - is_outlier)).unsqueeze(1),
                            dense_quantizer.scale,
                            dense_quantizer.zero,
                            dense_quantizer.maxq,
                        )
                        weight_i_quantized_wo_outliers = dequantize(
                            dense_weight_quant_i, dense_quantizer.scale, dense_quantizer.zero
                        ).reshape_as(dense_weight[:, column_index])
                        dense_weight_i_quantized = (
                            weight_i_quantized_wo_outliers * (1 - is_outlier) + dense_weight[:, column_index] * is_outlier
                        )  # [out_dim]

                        if save_quantization:
                            dense_save_quant_dict["outliers_matrix"][:, column_index] = dense_weight[:, column_index] * is_outlier

                        del weight_i_quantized_wo_outliers

                        delta_weight_i = dense_weight[:, column_index] - dense_weight_i_quantized  # [out_dim]
                        dense_quantization_errors[:, column_index] = (
                            delta_weight_i / dense_H_inv_cho[column_index, column_index]
                        )  # [out_dim]

                    if save_quantization:
                        dense_save_quant_dict["quant_weights"].append(dense_weight_quant_i.to(torch.int8))

                    dense_weight[:, column_index] = dense_weight_i_quantized
                    dense_weight[:, column_index + 1 : block_end].addr_(
                        dense_quantization_errors[:, column_index],
                        dense_H_inv_cho[column_index, column_index + 1 : block_end],
                        alpha=-1,
                    )

                dense_weight[:, block_end:].addmm_(
                    dense_quantization_errors[:, block_start:block_end],
                    dense_H_inv_cho[block_start:block_end, block_end:],
                    alpha=-1,
                )

            if permutation_order != "identity":
                invperm = torch.argsort(dense_perm)
                dense_weight = dense_weight[:, invperm]

            if save_quantization:
                dense_save_quant_dict["perm"] = dense_perm.to(torch.int32)
                dense_save_quant_dict["keep_last_columns"] = 0
                dense_save_quant_dict["blocksize"] = 128
                dense_save_quant_dict["weight_shape"] = sparse_weight.shape
                dense_save_quant_dict["groupsize"] = groupsize if groupsize else sparse_weight.shape[1]
                dense_save_quant_dict["quant_weights"] = torch.cat(dense_save_quant_dict["quant_weights"], dim=1)
                dense_save_quant_dict["outliers_matrix"] = dense_save_quant_dict["outliers_matrix"].to_sparse()

            if get_layer_size:
                bit_size = self.layer.sparse_weight.get_block_mask_size() + self.layer.sparse_weight.get_sparse_data_size()*sparse_bits + self.layer.sparse_weight.get_dense_data_size()*bits
                # gb_size = bit_size / 8 / 1024 / 1024 / 1024
                gb_size = bit_size

            result =  SparseQuantizationResult(
                sparse_weight=sparse_weight,
                dense_weight=dense_weight,
                sparse_perm=sparse_perm,
                dense_perm=dense_perm,
                sparse_quantization_errors=sparse_quantization_errors,
                dense_quantization_errors=dense_quantization_errors,
                sparse_unstructured_outlier_threshold=sparse_unstructured_outlier_threshold,
                dense_unstructured_outlier_threshold=dense_unstructured_outlier_threshold,
                sparse_unstructured_outlier_mask=sparse_unstructured_outlier_mask,
                dense_unstructured_outlier_mask=dense_unstructured_outlier_mask,
                sparse_save_quant_dict=sparse_save_quant_dict,
                dense_save_quant_dict=dense_save_quant_dict,
                layer_size = gb_size
            )
            
            return result


        # In the case there is no sparse and dense weights
        else:
            for block_start in block_start_iter:
                block_end = min(block_start + blocksize, in_dim)
                for column_index in range(block_start, block_end):
                    if column_index % groupsize == 0:
                        # fit weight quantizer on the upcoming group of weight columns (inputs), across all rows (outputs)
                        in_group_index += 1
                        group_weight = weight[:, column_index : column_index + groupsize]

                        if simplified_outliers or (unstructured_outlier_threshold == float("inf")):
                            quantizer.find_params(group_weight, weight=True)

                        else:
                            # objective: detect which weights will be designated as outliers, fit quantizer *without* these weights
                            # step 1: fit quantizer on a leave-one-out version of weights, i.e. in each group, drop one weight at a time
                            assert perchannel, "refitting quantizer is only implemented for perchannel=True"
                            group_diag_hessian_inv_cho = H_inv_cho_diag[column_index : column_index + groupsize]
                            loo_quantization_error_sq = get_leave_one_out_error(
                                group_weight, group_diag_hessian_inv_cho, bits=bits, sym=sym
                            )
                            # ^-- dequantized(quantized(group_weight)) using a quantizer trained on all weights except the reconstructed one

                            likely_unstructured_outlier_mask = (
                                loo_quantization_error_sq > unstructured_outlier_threshold
                            ).float()

                            non_outlier_mask = 1 - likely_unstructured_outlier_mask
                            mean_over_non_outliers = torch.sum(
                                group_weight * non_outlier_mask, dim=1, keepdim=True
                            ) / torch.sum(non_outlier_mask, dim=1, keepdim=True).clamp_min(1)
                            group_weight_without_outliers = group_weight * non_outlier_mask + mean_over_non_outliers * (
                                1 - non_outlier_mask
                            )
                            quantizer.find_params(group_weight_without_outliers, weight=True)
                            del group_diag_hessian_inv_cho, loo_quantization_error_sq
                            del mean_over_non_outliers, group_weight_without_outliers, non_outlier_mask

                        # Made a new method to make the save qunatization process easier.
                        if save_quantization:
                            quantizer, save_quant_dict = save_quantization(quantizer, save_quant_dict)
                        del group_weight
                        torch.cuda.empty_cache()

                    weight_quant_i = quantize(
                        weight[:, column_index].unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq
                    )
                    weight_i_quantized = dequantize(weight_quant_i, quantizer.scale, quantizer.zero).reshape_as(
                        weight[:, column_index]
                    )

                    delta_weight_i = weight[:, column_index] - weight_i_quantized  # [out_dim]
                    quantization_errors[:, column_index] = (
                        delta_weight_i / H_inv_cho[column_index, column_index]
                    )  # [out_dim]

                    if unstructured_outlier_threshold != float("inf"):
                        unstructured_outlier_mask[:, column_index] = (
                            quantization_errors[:, column_index].square() > unstructured_outlier_threshold
                        )
                        # re-quantize without outliers
                        is_outlier = unstructured_outlier_mask[:, column_index].float()

                    if save_quantization:
                        save_quant_dict["quant_weights"].append(weight_quant_i.to(torch.int8))

            if save_quantization:
                save_quant_dict["perm"] = perm.to(torch.int32)
                save_quant_dict["keep_last_columns"] = 0
                save_quant_dict["blocksize"] = 128
                save_quant_dict["weight_shape"] = weight.shape
                save_quant_dict["groupsize"] = groupsize if groupsize else weight.shape[1]
                save_quant_dict["quant_weights"] = torch.cat(save_quant_dict["quant_weights"], dim=1)
                save_quant_dict["outliers_matrix"] = save_quant_dict["outliers_matrix"].to_sparse()

            # Code for creating the block mask
            print("unstruc",unstructured_outlier_mask.sum().item())
            matrix_blocks = unstructured_outlier_mask.view(int(weight.shape[0]/block_shape[0]), block_shape[0], int(weight.shape[1]/block_shape[1]), block_shape[1])
            count_per_block = torch.sum(matrix_blocks, dim=(1, 3))
            
            block_mask = count_per_block > int(block_shape[0] * block_shape[1] * block_mask_density)
            block_mask = block_mask.view(int(weight.shape[0]/block_shape[0]), int(weight.shape[1]/block_shape[1]))
            
            print("max_blocks", block_mask.shape[0]*block_mask.shape[1])
            if torch.allclose(block_mask, torch.zeros_like(block_mask)):
                block_mask = count_per_block > int(block_shape[0] * block_shape[1] * block_mask_density * 0.5)
                block_mask = block_mask.view(int(weight.shape[0]/block_shape[0]), int(weight.shape[1]/block_shape[1]))
                if torch.allclose(block_mask, torch.zeros_like(block_mask)):
                    block_mask[0][0] = True
            print("block", block_mask.sum().item())

            if plot and layer_no in [0, 10, 23]:
                self.plot_mask(pdf, unstructured_outlier_mask, fig_title, outier_sensitivities=None)
                self.plot_mask(pdf, block_mask, fig_title+"_block_mask", outier_sensitivities=None)


            # Calculate the number of bits
            perc_out = (block_mask.sum().item()*block_shape[0]*block_shape[1]*sparse_bits)/(weight.shape[0]*weight.shape[1]*bits)*100
            return QuantizationResult(
                weight=weight,
                perm=perm,
                quantization_errors=quantization_errors,
                unstructured_outlier_threshold=unstructured_outlier_threshold,
                unstructured_outlier_mask=unstructured_outlier_mask,
                save_quant_dict=save_quant_dict,
                block_mask=block_mask,
                perc_out=perc_out,
            )


class QuantizationResult(NamedTuple):
    """A collection of codebooks, indices and assorted statistics produced by SPQRUtil; not memory-optimized!"""

    weight: torch.FloatTensor  # dequantized(quantized(weight)), same shape as the original
    perm: Optional[torch.LongTensor]  # optional input permutation indices that were used during quantization
    # NOTE: if permutation_order != identity, all subsequent tensors (incl. outlier indices) are permuted in that order!

    quantization_errors: torch.Tensor  # per-element quantization errors, defined as (weight - quantized_weight) / diag(inverse_hessian_cholesky)
    unstructured_outlier_threshold: float  # threshold on squared error increase used for determining *UNSTRUCTURED* outliers
    unstructured_outlier_mask: torch.Tensor  # bool mask where True means that this is an individual outlier
    save_quant_dict: dict
    block_mask: torch.Tensor
    perc_out: float

class SparseQuantizationResult(NamedTuple):
    """A collection of codebooks, indices and assorted statistics produced by SPQRUtil; not memory-optimized!"""

    sparse_weight: torch.FloatTensor  # dequantized(quantized(sparse_weight)), same shape as the original
    dense_weight: torch.FloatTensor  # dequantized(quantized(dense_weight)), same shape as the original
    sparse_perm: Optional[torch.LongTensor]  # optional input permutation indices that were used during quantization
    dense_perm: Optional[torch.LongTensor]  # optional input permutation indices that were used during quantization
    # NOTE: if permutation_order != identity, all subsequent tensors (incl. outlier indices) are permuted in that order!

    sparse_quantization_errors: torch.Tensor  # per-element quantization errors, defined as (weight - quantized_weight) / diag(inverse_hessian_cholesky)
    dense_quantization_errors: torch.Tensor  # per-element quantization errors, defined as (weight - quantized_weight) / diag(inverse_hessian_cholesky)
    sparse_unstructured_outlier_threshold: float  # threshold on squared error increase used for determining *UNSTRUCTURED* outliers
    dense_unstructured_outlier_threshold: float  # threshold on squared error increase used for determining *UNSTRUCTURED* outliers
    sparse_unstructured_outlier_mask: torch.Tensor  # bool mask where True means that this is an individual outlier
    dense_unstructured_outlier_mask: torch.Tensor  # bool mask where True means that this is an individual outlier
    sparse_save_quant_dict: dict
    dense_save_quant_dict: dict
    layer_size: float

def get_leave_one_out_error(group_weight: torch.Tensor, group_diag_hessian_inv_cho: torch.Tensor, *, bits, sym):
    """EXPERIMENTAL! BEWARE - for each weight, fit quantizer without this_one_weight and return this one weight's reconstruction"""

    assert group_weight.ndim == 2
    loo_indices = torch.arange(group_weight.shape[1], device=group_weight.device)
    loo_indices = loo_indices[1:] - (loo_indices[:, None] >= loo_indices[1:]).to(loo_indices.dtype)
    groupwise_loo_data = group_weight[:, loo_indices]  # [num_groups, num_loo = groupsize, groupsize - 1]
    fast_quantizer = Quantizer(shape=groupwise_loo_data.flatten(0, 1).shape)
    fast_quantizer.configure(bits, perchannel=True, sym=sym)
    fast_quantizer.find_params(groupwise_loo_data.flatten(0, 1), weight=True)

    # compute error improvement from not quantizing each one weight
    # to do so, we shall first train quantizer on leave-one-out data (which can be done faster since not all data affects quantization)
    loo_groupwise_reconstructed_weights = fast_quantizer.quantize_dequantize(
        groupwise_loo_data.flatten(0, 1)
    ).reshape_as(groupwise_loo_data)
    loo_group_diag_hessian_inv_cho = group_diag_hessian_inv_cho[loo_indices]  # [num_loo = groupsize, groupsize - 1]
    assert group_diag_hessian_inv_cho.ndim == 1

    # total quantization error consists of hessian-weighted mse on all remaining weights except for the one that's left out
    # -- this is because the left-out weights will not be quantized, and therefore, has zero quantization error
    loo_errors_sq = (
        ((loo_groupwise_reconstructed_weights - groupwise_loo_data) / loo_group_diag_hessian_inv_cho).square().sum(-1)
    )
    assert loo_errors_sq.shape == group_weight.shape  # [num_groups, num_loo = groupsize]

    # as a baseline error, quantize data normally without outliers
    base_quantizer = Quantizer(shape=group_weight.shape)
    base_quantizer.configure(bits, perchannel=True, sym=sym)
    base_quantizer.find_params(group_weight, weight=True)
    baseline_reconstructed_weights = base_quantizer.quantize_dequantize(group_weight)
    baseline_errors_sq = (
        ((baseline_reconstructed_weights - group_weight) / group_diag_hessian_inv_cho).square().sum(dim=1, keepdim=True)
    )

    # outlier's usefulness = how much does mse decrease from treating this weight as an outlier
    reduction_in_squared_error = baseline_errors_sq - loo_errors_sq
    return reduction_in_squared_error
