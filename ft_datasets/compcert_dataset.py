# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/compcert

import datasets
from .utils import Concatenator

def get_preprocessed_compcert(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset('csv', data_files={'test': '/home/ubuntu/llm-experiments/compcert_testing.csv', 'train': '/home/ubuntu/llm-experiments/compcert_training.csv'})[split]


    prompt = (
        f"{{theorem}}\n{{proof}}{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                theorem=sample["Theorem"].replace("Lemma", "Theorem"),
                proof=sample["Proof"],
                eos_token=tokenizer.eos_token,
            )
        }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
        
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset
