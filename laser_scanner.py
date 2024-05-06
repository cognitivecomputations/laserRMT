import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
from prompt_toolkit.shortcuts import checkboxlist_dialog, input_dialog
import argparse
from tqdm import tqdm
import os
import time

class ModelModifier:
    def __init__(self, model_name=None, top_percent=50, batch_size=1):
        self.model_name = model_name
        self.top_percent = top_percent
        self.batch_size = batch_size

        if model_name:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True, device_map="auto")
            self.optimizer = torch.optim.Adam(self.model.parameters())
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True, add_prefix_space=True)
        else:
            self.model = None
            self.optimizer = None
            self.tokenizer = None

        self.layer_snr = {}
        self.layer_types = []

    def get_weight_types(self):
        weight_types = set()
        for name, module in self.model.named_modules():
            parts = name.split('.')
            if any(hasattr(module, attr) for attr in ['weight', 'bias','inv_freq']):
                layer_index = next((i for i, part in enumerate(parts) if part.isdigit()), -1)
                weight_type = '.'.join(parts[layer_index + 1:]) if layer_index != -1 else name
                weight_types.add(weight_type)
        return list(weight_types)

    def interactive_select_weights(self):
        weight_types = self.get_weight_types()
        sorted_weight_types = self.sort_weight_types(weight_types)
        selected_types = checkboxlist_dialog(
            title="Select Weight Types", 
            text="Deselect the weight types you do not want to scan for SNR:",
            values=[(wt, wt) for wt in sorted_weight_types],
            default_values=sorted_weight_types
        ).run()
        self.layer_types = selected_types
        return selected_types

    def sort_weight_types(self, weight_types):
        categories = {}
        for wt in weight_types:
            category = wt.split('.')[0]
            categories.setdefault(category, []).append(wt)
        sorted_categories = {k: sorted(v) for k, v in sorted(categories.items(), key=lambda item: item[0])}
        sorted_weight_types = [wt for sublist in sorted_categories.values() for wt in sublist]
        return sorted_weight_types

    def calculate_snr_for_layer(self, layer_type):
        layers = [(name, module) for name, module in self.model.named_modules() if layer_type in name and hasattr(module, 'weight')]
        num_batches = (len(layers) + self.batch_size - 1) // self.batch_size

        with tqdm(total=num_batches, unit='batch', desc=f'Calculating SNR for {layer_type}') as progress_bar:
            for i in range(0, len(layers), self.batch_size):
                batch_layers = layers[i:i + self.batch_size]
                for name, module in batch_layers:
                    weights = module.weight.detach()
                    if weights.ndim < 2:
                        weights = weights.unsqueeze(0)
                    S = torch.linalg.svdvals(weights)
                    max_singular_value = S[0]
                    sigma_estimated = self.estimate_sigma_with_full_iqr(S)
                    n, m = weights.shape[-2:]
                    mp_threshold = self.marchenko_pastur_threshold(sigma_estimated, n, m)
                    signal = S[S > mp_threshold].sum()
                    noise = S[S <= mp_threshold].sum()
                    snr = signal / noise if noise != 0 else float('inf')
                    snr_ratio = snr / max_singular_value
                    self.layer_snr[name] = {'type': layer_type, 'snr': snr_ratio.item()}
                progress_bar.update(1)

    @staticmethod
    def marchenko_pastur_threshold(sigma, n, m):
        beta = n / m if n < m else m / n
        threshold = sigma * np.sqrt((1 + np.sqrt(beta)) ** 2)
        return threshold

    @staticmethod
    def estimate_sigma_with_full_iqr(S):
        q75 = torch.quantile(S, 0.75)
        q25 = torch.quantile(S, 0.25)
        iqr = q75 - q25
        sigma_estimated = iqr / 1.349
        return sigma_estimated

    def assess_layers_snr(self, selected_weight_types):
        total_layers = sum(1 for name, module in self.model.named_modules() if any(layer_type in name for layer_type in selected_weight_types) and hasattr(module, 'weight'))
        start_time = time.time()

        with tqdm(total=len(selected_weight_types), unit='type', desc='Calculating SNR for types') as progress_bar:
            for layer_type in selected_weight_types:
                self.calculate_snr_for_layer(layer_type)
                progress_bar.update(1)

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time taken: {total_time:.2f} seconds")

    def save_snr_to_json(self):
        filename = f"snr_results_{self.model_name.split('/')[-1]}.json" if self.model_name else "snr_results.json"
        serializable_data = {}
        for layer_name, info in self.layer_snr.items():
            snr_value = info['snr'].item() if isinstance(info['snr'], torch.Tensor) else info['snr']
            layer_type = str(info['type'])
            serializable_data[layer_name] = {'snr': snr_value, 'type': layer_type}
        with open(filename, 'w') as file:
            json.dump(serializable_data, file, indent=4)
        print(f"Results saved to {filename}")
        self.save_top_snr_ratios_to_json(filename)
        self.generate_unfrozen_params_yaml(filename)

    def generate_unfrozen_params_yaml(self, json_filename, top_percent=None):
        top_percent = top_percent if top_percent is not None else self.top_percent
        with open(json_filename, 'r') as file:
            snr_data = json.load(file)
        unfrozen_parameters = {}
        for layer_name, info in snr_data.items():
            layer_type = info['type']
            if layer_type not in unfrozen_parameters:
                unfrozen_parameters[layer_type] = []
            unfrozen_parameters[layer_type].append((layer_name, info['snr']))
        top_layers_by_type = {}
        for layer_type, layers in unfrozen_parameters.items():
            layers_sorted = sorted(layers, key=lambda x: x[1], reverse=True)
            num_top_layers = int(len(layers) * top_percent / 100)
            top_layers_by_type[layer_type] = [layer[0] for layer in layers_sorted[:num_top_layers]]
        # Modify the yaml_filename to include the input json name and top_percent
        json_file_base = os.path.splitext(os.path.basename(json_filename))[0]
        yaml_filename = f"{json_file_base}_unfrozenparameters_{top_percent}percent.yaml"
        with open(yaml_filename, 'w') as file:
            file.write("unfrozen_parameters:\n")
            for layer_type, layer_names in top_layers_by_type.items():
                file.write(f"# {layer_type} layers\n")
                for layer_name in layer_names:
                    file.write(f"- {layer_name}\n")
        print(f"Top {top_percent}% SNR layers saved to {yaml_filename}")

    def save_top_snr_ratios_to_json(self, json_filename, filename=None):
        with open(json_filename, 'r') as file:
            snr_data = json.load(file)
        all_snr_layers = {}
        for layer_name, info in snr_data.items():
            layer_type = info['type']
            if layer_type not in all_snr_layers:
                all_snr_layers[layer_type] = []
            all_snr_layers[layer_type].append((layer_name, info['snr']))
        for layer_type, layers in all_snr_layers.items():
            layers_sorted = sorted(layers, key=lambda x: x[1], reverse=True)
            all_snr_layers[layer_type] = {layer[0]: layer[1] for layer in layers_sorted}

        json_file_base = os.path.splitext(os.path.basename(json_filename))[0]
        filename = f"{json_file_base}_sorted.json" if filename is None else filename

        with open(filename, 'w') as file:
            json.dump(all_snr_layers, file, indent=4)
        print(f"All SNR layers sorted and saved to {filename}")

# Handle command-line arguments
parser = argparse.ArgumentParser(description="Process SNR data for layers.")
parser.add_argument('--json', type=str, help='Path to existing JSON file for processing')
parser.add_argument('--top_percent', type=int, default=None, help='Top percentage of layers to select, overriding the default')
args = parser.parse_args()

if args.json:
    modifier = ModelModifier(top_percent=args.top_percent)
    modifier.generate_unfrozen_params_yaml(args.json, args.top_percent)
else:
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    batch_size = input_dialog(title="Batch Size", text="Enter the batch size:").run()
    batch_size = int(batch_size) if batch_size else 1
    modifier = ModelModifier(model_name=model_name, batch_size=batch_size)
    selected_weight_types = modifier.interactive_select_weights()
    if selected_weight_types:
        modifier.assess_layers_snr(selected_weight_types)
        modifier.save_snr_to_json()
        print("Finished SNR scanning and data saved.")
    else:
        print("No weight types selected.")
