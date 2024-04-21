import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import gc
import json
from prompt_toolkit.shortcuts import checkboxlist_dialog

class ModelModifier:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
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
        batch_size = self.determine_batch_size(layer_type)
        layers = [(name, module) for name, module in self.model.named_modules() if layer_type in name and hasattr(module, 'weight')]
        num_batches = (len(layers) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(layers))
            batch_layers = layers[start_idx:end_idx]

            for name, module in batch_layers:
                weights = module.weight.detach().double()
                S = torch.linalg.svdvals(weights)
                max_singular_value = S[0].item()
                sigma_estimated = self.estimate_sigma_with_full_iqr(S)
                n, m = weights.shape
                mp_threshold = self.marchenko_pastur_threshold(sigma_estimated, n, m)
                signal = S[S > mp_threshold].sum().item()
                noise = S[S <= mp_threshold].sum().item()
                snr = signal / noise if noise != 0 else float('inf')
                snr_ratio = snr / max_singular_value
                self.layer_snr[name] = snr_ratio
                print(f"Calculated SNR for {name}: {snr_ratio}")

            del S, weights
            gc.collect()
            torch.cuda.empty_cache()

    def determine_batch_size(self, layer_type):
        max_batch_size = len([(name, module) for name, module in self.model.named_modules() if layer_type in name and hasattr(module, 'weight')])
        batch_size = max_batch_size

        while True:
            try:
                layers = [(name, module) for name, module in self.model.named_modules() if layer_type in name and hasattr(module, 'weight')]
                batch_layers = layers[:batch_size]

                for name, module in batch_layers:
                    weights = module.weight.detach().double()
                    S = torch.linalg.svdvals(weights)
                    del S, weights

                gc.collect()
                torch.cuda.empty_cache()
                return batch_size

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    batch_size //= 2
                    if batch_size == 0:
                        raise RuntimeError("Batch size cannot be further reduced. Insufficient VRAM.")
                else:
                    raise e

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
        sorted_layer_snr = dict(sorted(self.layer_snr.items(), key=lambda x: x[1], reverse=True))
        with open(filename, 'w') as file:
            json.dump({k: float(v) for k, v in sorted_layer_snr.items()}, file, indent=4)
        print(f"Results saved to {filename}")

# Usage
model_name = "01-ai/Yi-34B-200K"
modifier = ModelModifier(model_name)
selected_weight_types = modifier.interactive_select_weights()

if selected_weight_types:
    modifier.assess_layers_snr(selected_weight_types)
    modifier.save_snr_to_json()
    print("Finished SNR scanning and data saved.")
else:
    print("No weight types selected.")
