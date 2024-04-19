import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import gc
import json
from accelerate import Accelerator
from prompt_toolkit.shortcuts import checkboxlist_dialog

class ModelModifier:
    def __init__(self, model_name):
        self.accelerator = Accelerator()
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.model = self.accelerator.prepare(self.model)  # Prepare the model for multi-GPU
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_prefix_space=True)
        self.layer_snr = {}

    def get_weight_types(self):
        weight_types = set()
        for name, module in self.model.named_modules():
            parts = name.split('.')
            if hasattr(module, 'weight') and len(parts) > 2:
                weight_types.add(parts[-1])
        return list(weight_types)

    def interactive_select_weights(self):
        weight_types = self.get_weight_types()
        selected_types = checkboxlist_dialog(
            title="Select Weight Types",
            text="Select weight types to scan for SNR:",
            values=[(wt, wt) for wt in weight_types]
        ).run()
        return selected_types

    def calculate_snr_for_layer(self, layer_type):
        batch_size = self.determine_batch_size()
        layers = [(name, module) for name, module in self.model.named_modules() if layer_type in name and hasattr(module, 'weight')]
        for i in range(0, len(layers), batch_size):
            batch_layers = layers[i:i + batch_size]
            self.accelerator.wait_for_everyone()  # Ensure all processes are synchronized before processing
            for name, module in batch_layers:
                weights = module.weight.detach().double()
                S = torch.linalg.svdvals(weights)
                max_singular_value = S[0].item()
                sigma_estimated = self.estimate_sigma_with_full_iqr(S)
                n, m = weights.shape
                mp_threshold = self.marchenko_pastur_threshold(sigma_estimated, n, m)
                signal = S[S > mp_threshold].sum()
                noise = S[S <= mp_threshold].sum()
                snr = signal / noise if noise != 0 else float('inf')
                snr_ratio = snr / max_singular_value
                self.layer_snr[name] = snr_ratio
                print(f"Calculated SNR for {name}: {snr_ratio}")
                del S, weights
                gc.collect()
            self.accelerator.wait_for_everyone()  # Synchronize after batch completion

    def determine_batch_size(self):
        # This should be dynamically calculated based on available GPU memory across devices
        total_memory = torch.cuda.get_device_properties(self.accelerator.device).total_memory
        free_memory = total_memory - torch.cuda.memory_reserved(self.accelerator.device)
        estimated_memory_per_layer = 512 * 1024 * 1024  # Estimate 512 MB per layer
        return max(1, int(free_memory / estimated_memory_per_layer))

    @staticmethod
    def marchenko_pastur_threshold(sigma, n, m):
        beta = n / m if n < m else m / n
        threshold = sigma * np.sqrt((1 + np.sqrt(beta))**2)
        return threshold

    @staticmethod
    def estimate_sigma_with_full_iqr(S):
        q75 = torch.quantile(S, 0.75)
        q25 = torch.quantile(S, 0.25)
        iqr = q75 - q25
        sigma_estimated = iqr / 1.349
        return sigma_estimated

    def assess_layers_snr(self, selected_weight_types):
        for layer_type in selected_weight_types:
            self.calculate_snr_for_layer(layer_type)

    def save_snr_to_json(self):
        filename = f"snr_results_{self.model_name.split('/')[-1]}.json"
        with open(filename, 'w') as file:
            json.dump(self.layer_snr, file, indent=4)
            print(f"Results saved to {filename}")

# Usage
model_name = "mistral-community/Mistral-7B-v0.2"
modifier = ModelModifier(model_name)
selected_weight_types = modifier.interactive_select_weights()
if selected_weight_types:
    modifier.assess_layers_snr(selected_weight_types)
    modifier.save_snr_to_json()
    print("Finished SNR scanning and data saved.")
else:
    print("No weight types selected.")
