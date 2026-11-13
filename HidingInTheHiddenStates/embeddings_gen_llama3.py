import os
os.environ['HF_TOKEN'] = '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'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import numpy as np
from typing import Dict

path_to_model = "meta-llama/Llama-3.2-3B"
model_to_use = "llama3.2_3b"

layers_to_use = [-1, -4, -8, -12, -16]
list_of_datasets = [
    "capitals",
    "inventions",
    "elements",
    "animals",
    "companies",
    "facts"
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
remove_period = True

print(f"Loading {path_to_model} on {device}...")
model = AutoModelForCausalLM.from_pretrained(path_to_model, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(path_to_model)
print("Model loaded successfully!")

dfs: Dict[int, pd.DataFrame] = {}

with torch.no_grad():
    for dataset_to_use in list_of_datasets:
        print(f"Processing dataset: {dataset_to_use}")
        df = pd.read_csv("resources/" + dataset_to_use + "_true_false.csv")
        df['embeddings'] = pd.Series(dtype='object')
        df['next_id'] = pd.Series(dtype=float)
        for layer in layers_to_use:
            dfs[layer] = df.copy()

        for i, row in df.iterrows():
            prompt = row['statement']
            if remove_period:
                prompt = prompt.rstrip(". ")
            inputs = tokenizer(prompt, return_tensors="pt")

            outputs = model.generate(inputs.input_ids.to(device), output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=1, min_new_tokens=1)
            generate_ids = outputs[0].cpu()
            next_id = np.array(generate_ids)[0][-1]
            for layer in layers_to_use:
                last_hidden_state = outputs.hidden_states[0][layer][0][-1].cpu()
                dfs[layer].at[i, 'embeddings'] = [last_hidden_state.numpy().tolist()]
                dfs[layer].at[i, 'next_id'] = next_id
            if i % 100 == 0 or i == len(df) - 1:
                print(f"processing: {i}/{len(df)}, next_token: {next_id}")

        for layer in layers_to_use:
            out_path = "embeddings/" + "embeddings_with_labels_" + dataset_to_use + model_to_use + "_" + str(abs(layer)) + "_rmv_period.csv"
            dfs[layer].to_csv(out_path, index=False)
            print(f"Saved {out_path}")

print("All embeddings generated successfully!")
