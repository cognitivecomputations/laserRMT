# Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
from prompt_toolkit.shortcuts import checkboxlist_dialog
import argparse
from tqdm import tqdm

# Define a class to modify model parameters
class ModelModifier:
    def __init__(self, model_name):
        # Initialize the model with specific configurations if model name is provided
        if model_name:
            self.model_name = model_name
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True, device_map="auto")
            self.optimizer = torch.optim.Adam(self.model.parameters())
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True, add_prefix_space=True)
        else:
            # Set model related attributes to None if no model name is provided
            self.model_name = None
            self.model = None
            self.optimizer = None
            self.tokenizer = None
        self.layer_snr = {}
        self.layer_types = []

    def get_weight_types(self):
        # Identify unique weight types in the model
        weight_types = set()
        for name, module in self.model.named_modules():
            parts = name.split('.')
            if hasattr(module, 'weight') and len(parts) > 2:
                weight_types.add(parts[-1])
        return list(weight_types)

    def interactive_select_weights(self):
        # User interface to select weight types to analyze with all options pre-selected by default
        weight_types = self.get_weight_types()
        selected_types = checkboxlist_dialog(
            title="Select Weight Types",
            text="Deselect the weight types you do not want to scan for SNR:",
            values=[(wt, wt) for wt in weight_types] 
        ).run()
        self.layer_types = selected_types
        return selected_types

    def calculate_snr_for_layer(self, layer_type):
        # Calculate signal-to-noise ratio for each layer of a selected type
        batch_size = 3
        layers = [(name, module) for name, module in self.model.named_modules() if layer_type in name and hasattr(module, 'weight')]
        num_batches = (len(layers) + batch_size - 1) // batch_size
        progress_bar = tqdm(total=num_batches, unit='batch', desc=f'Calculating SNR for {layer_type}')

        for i in range(0, len(layers), batch_size):
            batch_layers = layers[i:i + batch_size]
            for name, module in batch_layers:
                weights = module.weight.detach()
                if weights.ndim < 2:
                    weights = weights.unsqueeze(0)
                S = torch.linalg.svdvals(weights)
                max_singular_value = S[0].item()
                sigma_estimated = self.estimate_sigma_with_full_iqr(S)
                n, m = weights.shape[-2:]
                mp_threshold = self.marchenko_pastur_threshold(sigma_estimated, n, m)
                signal = S[S > mp_threshold].sum().item()
                noise = S[S <= mp_threshold].sum().item()
                snr = signal / noise if noise != 0 else float('inf')
                snr_ratio = snr / max_singular_value
                self.layer_snr[name] = {'type': layer_type, 'snr': snr_ratio}
            progress_bar.update(1)
        progress_bar.close()

    @staticmethod
    def marchenko_pastur_threshold(sigma, n, m):
        # Calculate the Marchenko-Pastur threshold for signal identification
        beta = n / m if n < m else m / n
        threshold = sigma * np.sqrt((1 + np.sqrt(beta)) ** 2)
        return threshold

    @staticmethod
    def estimate_sigma_with_full_iqr(S):
        # Estimate noise level using the interquartile range
        q75 = torch.quantile(S, 0.75)
        q25 = torch.quantile(S, 0.25)
        iqr = q75 - q25
        sigma_estimated = iqr / 1.349
        return sigma_estimated

    def assess_layers_snr(self, selected_weight_types):
        # Assess SNR for selected layer types
        for layer_type in selected_weight_types:
            self.calculate_snr_for_layer(layer_type)

    def save_snr_to_json(self):
        # Save SNR results to a JSON file
        filename = f"snr_results_{self.model_name.split('/')[-1]}.json" if self.model_name else "snr_results.json"
        type_groups = {type_: [] for type_ in self.layer_types}
        for layer_name, info in self.layer_snr.items():
            type_groups[info['type']].append({layer_name: info['snr']})
        sorted_type_groups = {}
        for type_name, layers in type_groups.items():
            sorted_type_groups[type_name] = sorted(layers, key=lambda x: list(x.values())[0], reverse=True)
        with open(filename, 'w') as file:
            json.dump(sorted_type_groups, file, indent=4)
        print(f"Results saved to {filename}")
        self.generate_unfrozen_params_yaml(sorted_type_groups)
    
    def generate_unfrozen_params_yaml(self, sorted_type_groups, top_percent=50):
        # Generate a YAML file listing top n% layers with the highest SNR for each weight type
        unfrozen_parameters = []
        for type_name, type_layers in sorted_type_groups.items():
            num_layers = len(type_layers)
            num_top_layers = int(num_layers * top_percent / 100)
            top_layers = type_layers[:num_top_layers]
            for layer_dict in top_layers:
                layer_name = list(layer_dict.keys())[0]
                unfrozen_parameters.append(layer_name)

        with open(f"unfrozen_parameters_{self.model_name.split('/')[-1]}.yaml" if self.model_name else "unfrozen_parameters.yaml", 'w') as file:
            file.write("unfrozen_parameters:\n")
            for layer_name in unfrozen_parameters:
                file.write(f"- {layer_name}\n")

# Handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str, help='Path to existing JSON file')
args = parser.parse_args()

if args.json:
    # Load existing SNR data and generate parameters from it
    with open(args.json, 'r') as file:
        sorted_type_groups = json.load(file)
    modifier = ModelModifier("")  # Empty model name for JSON loading
    modifier.generate_unfrozen_params_yaml(sorted_type_groups, top_percent=50)
else:
    # Standard operation: model initialization and SNR analysis
    model_name = "Qwen/Qwen1.5-1.8B"
    modifier = ModelModifier(model_name)
    selected_weight_types = modifier.interactive_select_weights()
    if selected_weight_types:
        modifier.assess_layers_snr(selected_weight_types)
        modifier.save_snr_to_json()
        print("Finished SNR scanning and data saved.")
    else:
        print("No weight types selected.")
