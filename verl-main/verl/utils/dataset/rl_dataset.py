# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import ListConfig
import os
from typing import List, Union, Optional, Callable
import copy
import datasets
from collections import defaultdict
import base64

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: str, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    # image = Image.open(BytesIO(base64.b64decode(image)))
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key: str = 'prompt',
                 image_key: str = 'images',
                 max_prompt_length: int = 1024,
                 cache_dir: str = '~/.cache/verl/rlhf',
                 chat_template_func: Optional[Callable] = None,
                 return_raw_chat: bool = False,
                 truncation: str = 'error',
                 filter_overlong_prompts: bool = False,
                 num_workers: Optional[int] = None):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts
        if num_workers is None:
            self.num_workers = max(1, os.cpu_count() // 4)
        else:
            self.num_workers = min(num_workers, os.cpu_count())

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f'dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)
                               ) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens")

            print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)
        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            last_user_message: str = messages.pop(-1)['content']
            for i, content in enumerate(last_user_message.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            messages.append({"role": "user", "content": content_list})

        return messages
    
    # def __getitem__(self, item):
    #     """
    #     Note that we also return the raw_input_ids so that it can be combined with other chat template
    #     """
    #     row_dict: dict = self.dataframe[item]

    #     messages = self._build_messages(row_dict)
    #     # chat = row_dict.pop(self.prompt_key)
    #     is_multi_modal = self.image_key in row_dict
        
    #     if is_multi_modal:  # expand image token
    #         prompt_with_chat_template = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    #         raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
    #         row_dict['multi_modal_data'] = {'image': [process_image(image) for image in row_dict.pop(self.image_key)]}
    #         image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
    #         image_grid_thw = image_inputs['image_grid_thw']
    #         row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

    #         if image_grid_thw is not None:
    #             merge_length = self.processor.image_processor.merge_size**2
    #             index = 0
    #             while '<image>' in prompt_with_chat_template:
    #                 prompt_with_chat_template = prompt_with_chat_template.replace(
    #                     '<image>',
    #                     '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
    #                     '<|vision_end|>',
    #                     1,
    #                 )
    #                 index += 1

    #             prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
    #                                                                           self.processor.image_token)
    #     else:
    #         prompt_with_chat_template = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    #         raw_prompt = prompt_with_chat_template

    #     input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
    #                                                                      tokenizer=self.tokenizer,
    #                                                                      max_length=self.max_prompt_length,
    #                                                                      pad_token_id=self.tokenizer.pad_token_id,
    #                                                                      left_pad=True,
    #                                                                      truncation=self.truncation)

    #     if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
    #         from verl.models.transformers.qwen2_vl import get_rope_index

    #         position_ids = get_rope_index(
    #             self.processor,
    #             input_ids=input_ids[0],
    #             image_grid_thw=image_grid_thw,
    #             attention_mask=attention_mask[0],
    #         )  # (3, seq_len)
    #     else:
    #         position_ids = compute_position_id_with_mask(attention_mask)

    #     row_dict['input_ids'] = input_ids[0]
    #     row_dict['attention_mask'] = attention_mask[0]
    #     row_dict['position_ids'] = position_ids[0]
    #     row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

    #     # encode prompts without chat template
    #     if self.return_raw_chat:
    #         row_dict['raw_prompt'] = chat

    #     # add index for each prompt
    #     index = row_dict.get("extra_info", {}).get("index", 0)
    #     row_dict["index"] = index

    #     return row_dict

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """

        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        if self.image_key in row_dict:  # process multimodal data
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = [process_image(image) for image in row_dict.pop(self.image_key)]
            model_inputs = self.processor(images, [raw_prompt], return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop('input_ids')
            attention_mask = model_inputs.pop('attention_mask')
            row_dict["multi_modal_data"] = {"image": images}
            row_dict["multi_modal_inputs"] = dict(model_inputs)
        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors='pt', add_special_tokens=False)
            input_ids = model_inputs.pop('input_ids')
            attention_mask = model_inputs.pop('attention_mask')

        input_ids, attention_mask = verl_F.postprocess_data(input_ids=input_ids,
                                                            attention_mask=attention_mask,
                                                            max_length=self.max_prompt_length,
                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                            left_pad=True,
                                                            truncation=self.truncation)

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict['raw_prompt_ids'] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = messages

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
