from functools import lru_cache
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import pandas as pd
from icecream import ic
import random

default_context_length = 1024 * 8


class DatasetType(str, Enum):
    dpo = "dpo"
    binary = "binary"


class BinarySample(BaseModel):
    instruction: Optional[str]
    prompt: str
    chosen: str


class DPOSample(BaseModel):
    instruction: Optional[str]
    prompt: str
    chosen: str
    rejected: str


class Dataset(BaseModel):
    id: Union[str, int]
    name: str
    description: Optional[str] = None
    dataset_type: DatasetType
    has_system_instruction: bool
    data: Union[List[BinarySample], List[DPOSample]]


@lru_cache
def get_orca_dpo(n_samples, input_length, output_length) -> Dataset:
    from datasets import load_dataset

    dataset_id = "Intel/orca_dpo_pairs"
    testdata = load_dataset(dataset_id, split="train")
    testdata_filtered = (
        testdata.filter(
            lambda example: (
                len(example["system"]) + len(example["question"]) <= input_length
            )
            and (len(example["chosen"]) <= output_length)
            and (len(example["rejected"]) <= output_length)
        )
        # .shuffle(seed=random.randint(0, 10000))
        .select(range(n_samples))
    )

    return Dataset(
        id=dataset_id,
        name=dataset_id,
        dataset_type=DatasetType.dpo,
        has_system_instruction=True,
        data=[
            DPOSample(
                instruction=data["system"],
                prompt=data["question"],
                chosen=data["chosen"],
                rejected=data["rejected"],
            )
            for data in testdata_filtered
        ],
    )


@lru_cache
def get_ultrafeedback(n_samples, input_length, output_length) -> Dataset:
    from datasets import load_dataset

    dataset_id = "allenai/ultrafeedback_binarized_cleaned"
    testdata = load_dataset(dataset_id, split="test_prefs")

    testdata_filtered = (
        testdata.filter(
            lambda example: (len(example["prompt"]) <= input_length)
            and (len(example["chosen"][0]["content"]) <= output_length)
            and (len(example["rejected"][0]["content"]) <= output_length)
        )
        # .shuffle(seed=random.randint(0, 10000))
        .select(range(n_samples))
    )

    return Dataset(
        id=dataset_id,
        name=dataset_id,
        dataset_type=DatasetType.dpo,
        has_system_instruction=False,
        data=[
            DPOSample(
                instruction="",
                prompt=data["prompt"],
                chosen=data["chosen"][0]["content"],
                rejected=data["rejected"][0]["content"],
            )
            for data in testdata_filtered
        ],
    )


@lru_cache
def get_openhermes(n_samples, input_length, output_length) -> Dataset:
    from datasets import load_dataset

    dataset_id = "teknium/openhermes"

    testdata = load_dataset(dataset_id, split="train")

    testdata_filtered = (
        testdata.filter(
            lambda example: (
                len(example["instruction"]) + len(example["input"]) <= input_length
            )
            and (len(example["output"]) <= output_length)
        )
        # TODO shuffle actually good?
        # .shuffle(seed=random.randint(0, 10000))
        .select(range(n_samples))
    )

    return Dataset(
        id=dataset_id,
        name=dataset_id,
        dataset_type=DatasetType.binary,
        has_system_instruction=False,
        data=[
            BinarySample(
                instruction=data["instruction"],
                prompt=data["input"],
                chosen=data["output"],
            )
            for data in testdata_filtered
        ],
    )


@lru_cache
def get_benchmark_data(
    name,
    n_samples=256,
    input_length=default_context_length / 2,
    output_length=default_context_length / 2,
):
    if name == "orca_dpo":
        return get_orca_dpo(n_samples, input_length, output_length)
    elif name == "ultrafeedback":
        return get_ultrafeedback(n_samples, input_length, output_length)
    elif name == "openhermes":
        return get_openhermes(n_samples, input_length, output_length)

    else:
        raise Exception
