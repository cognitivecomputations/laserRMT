# %%
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Change to your preferred model

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from lib.utils import gptq_data_utils_math
from tqdm import tqdm
import random
import numpy as np
import gc
import time

class ModelModifier:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("mps:0")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.layer_snr = {}
        self.modified_layers = set()
        self.original_weights = {}
        
    def calculate_snr_for_layer(self, layer_type, layer_number):
        for name, module in self.model.named_modules():
            if layer_type in name and f"model.layers.{layer_number}." in name:
                weights = module.weight.float()
                S = torch.linalg.svdvals(weights)
                weights = weights.detach().cpu()
                S = S.detach().cpu()
                sigma_estimated = self.estimate_sigma_with_full_iqr(S)
                n, m = weights.shape
                mp_threshold = self.marchenko_pastur_threshold(sigma_estimated, n, m)

                signal = S[S > mp_threshold].sum()
                noise = S[S <= mp_threshold].sum()
                snr = signal / noise if noise != 0 else float('inf')
                del S, weights, signal, noise
                torch.mps.empty_cache()  # Clear PyTorch's MPS memory cache
                gc.collect()
                return snr

    def update_model_reduce_layer(self, layer_type, layer_number):
        layer_id = f"{layer_type}_{layer_number}"
        if layer_id in self.modified_layers:
            print(f"Layer {layer_id} has already been modified. Skipping.")
            return False

        for name, module in self.model.named_modules():
            if layer_type in name and f"model.layers.{layer_number}." in name:
                print(f"Reconstructing layer: {name}")
                original_dtype = module.weight.dtype
                self.original_weights[name] = module.weight.detach().clone()
                weights = module.weight.float()
                U, S, V = torch.linalg.svd(weights, full_matrices=False)

                # Estimate sigma using the full IQR method
                sigma_estimated_full_iqr = self.estimate_sigma_with_full_iqr(S)

                # Calculate Marchenko-Pastur threshold
                n, m = weights.shape
                mp_threshold_full_iqr = self.marchenko_pastur_threshold(sigma_estimated_full_iqr, n, m)

                # Retain only the singular values above the MP threshold
                S_reduced = torch.zeros_like(S)
                k = (S > mp_threshold_full_iqr).sum().item()
                S_reduced[:k] = S[:k]
                print(f"Reduced from {S.shape} to {k}")

                # Reconstruct the matrix using the thresholded singular values
                reconstructed_weights = U @ torch.diag(S_reduced) @ V
                reconstructed_weights = reconstructed_weights.to(original_dtype)
                module.weight = torch.nn.Parameter(reconstructed_weights)
                self.modified_layers.add(layer_id)

                del U, S, V, weights
                torch.mps.empty_cache()  # Clear PyTorch's MPS memory cache
                gc.collect()
                return True

    @staticmethod
    def marchenko_pastur_threshold(sigma, n, m):
        beta = n / m if n < m else m / n
        threshold = sigma * np.sqrt((1 + np.sqrt(beta))**2)
        return threshold
    
    ## Calculate an estimate of the standard deviation of the singular values based on Inter Quantile Range

    @staticmethod
    def estimate_sigma_with_full_iqr(S):
        q75 = torch.quantile(S, 0.75)
        q25 = torch.quantile(S, 0.25)
        iqr = q75 - q25
        sigma_estimated = iqr / 1.349 ## 0.6745 * sigma is the expected range between the quantiles (Q1 and Q3)
        return sigma_estimated
    
    def scanlayers(self):
        for name, module in self.model.named_modules():
            print(name, flush=True)

    def restore_model_original_layer(self, layer_type, layer_number):
        layer_id = f"{layer_type}_{layer_number}"
        for name, module in self.model.named_modules():
            if layer_type in name and f"model.layers.{layer_number}." in name:
                if name in self.original_weights:
                    module.weight = torch.nn.Parameter(self.original_weights[name])
                    print(f"Restored original weights for layer: {name}")
                    if layer_id in self.modified_layers:
                        self.modified_layers.remove(layer_id)
                else:
                    print(f"No original weights saved for layer: {name}")

    def calculate_model_perplexity(self, datasets=['gsm8k'], seqlen=32, use_cuda_graph=False, use_flash_attn=False):
        model = self.model
        model_str = self.model_name
        acc_loss = 0.0
        total_samples = 0

        for dataset in datasets:
            input_tok = gptq_data_utils_math.get_test_tokens(dataset, seed=0, seqlen=seqlen, model=model_str).to("mps:0")
            total_length = input_tok.size(0)
            nsamples = total_length // seqlen
            rest = total_length % seqlen

            if rest != 0:
            # if the last part of the data is not complete, we cut it off
                input_tok = input_tok[:-rest]

            input_tok = input_tok.view(-1, seqlen)  # reshape the tensor
            total_samples += nsamples

            #if not use_cuda_graph:
            #    model.reset()

            loss_fct = torch.nn.CrossEntropyLoss()
            progress = tqdm(range(nsamples))
            for ii in progress:
                input = input_tok[ii, :].view(1, -1)
                output = model(input, use_cache=False, output_hidden_states=False, output_attentions=False)[0]
                shift_logits = output[:, :-1, :].contiguous()
                shift_labels = input[:, 1:]
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                acc_loss += loss.item()
                progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

        avg_loss = acc_loss / total_samples
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        return ppl
    


    def assess_layers_snr(self, layer_types, layer_numbers):
        for name, _ in self.model.named_modules():
            for layer_number in layer_numbers:
                for layer_type in layer_types:
                    if layer_type in name and f"model.layers.{layer_number}." in name:
                        layer_name = f"{layer_type}+{layer_number}"
                        print("*"*50, flush=True)
                        print(f"Calculating Signal to Noise Ratio at layer {layer_name}", flush=True)
                        snr = self.calculate_snr_for_layer(layer_type, layer_number)
                        self.layer_snr[layer_name] = snr
                        print(f"Signal to Noise Ratio at layer {layer_name} = {snr}", flush=True)
                        print("*"*50, flush=True)

    def select_layers_for_modification(self, k):
        sorted_layers = sorted(self.layer_snr.items(), key=lambda x: x[1], reverse=False)
        return [layer[0] for layer in sorted_layers[:k]]
    
    def test_and_modify_layers(self, candidate_layers):
        initial_perplexity = self.calculate_model_perplexity()
        print(f"Initial Model Perplexity: {initial_perplexity}")

        for layer in candidate_layers:
            # Modify the layer
            layer_type = layer.split("+")[0]
            layer_number = layer.split("+")[1]
            self.update_model_reduce_layer(layer_type=layer_type,layer_number=layer_number)
            
            # Test the model's performance
            new_perplexity = self.calculate_model_perplexity()
            print(f"Tested Model Perplexity after modifying {layer}: {new_perplexity}")

            # If the perplexity does not improve significantly, revert the change
            if new_perplexity > initial_perplexity:
                self.restore_model_original_layer(layer_type=layer_type,layer_number=layer_number)
                print(f"Reverted changes in {layer} due to lack of improvement.", flush=True)
            else:
                initial_perplexity = new_perplexity
                print(f"Modification kept for {layer}. New baseline perplexity: {initial_perplexity}", flush=True)

    def save_model(self, save_dir):

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

# Usage
modifier = ModelModifier(model_name)

# %%
layer_numbers = list(range(31, -1, -1))
print(layer_numbers)

modifier.scanlayers()

#layer_types=['mlp.gate_proj','mlp.down_proj', 'mlp.up_proj', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
layer_types=['mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 
             'block_sparse_moe.experts.0.w1', 'block_sparse_moe.experts.0.w2', 'block_sparse_moe.experts.0.w3',
             'block_sparse_moe.experts.1.w1', 'block_sparse_moe.experts.1.w2', 'block_sparse_moe.experts.1.w3',
             'block_sparse_moe.experts.2.w1', 'block_sparse_moe.experts.2.w2', 'block_sparse_moe.experts.2.w3',
             'block_sparse_moe.experts.3.w1', 'block_sparse_moe.experts.3.w2', 'block_sparse_moe.experts.3.w3',
             'block_sparse_moe.experts.4.w1', 'block_sparse_moe.experts.4.w2', 'block_sparse_moe.experts.4.w3',
             'block_sparse_moe.experts.5.w1', 'block_sparse_moe.experts.5.w2', 'block_sparse_moe.experts.5.w3',
             'block_sparse_moe.experts.6.w1', 'block_sparse_moe.experts.6.w2', 'block_sparse_moe.experts.6.w3',
             'block_sparse_moe.experts.7.w1', 'block_sparse_moe.experts.7.w2', 'block_sparse_moe.experts.7.w3',]

modifier.assess_layers_snr(layer_types, layer_numbers)

top_k_layers = modifier.select_layers_for_modification(15)  # Select top 15 layers
print(top_k_layers, flush=True)

modifier.test_and_modify_layers(top_k_layers)
# %%
modifier.save_model("laser_model")


