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
"""
Preprocess the virl39k dataset to parquet format
"""

from datasets import Dataset, load_dataset, load_from_disk
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-hf-dataset', default='/path/to/hf_data')
    parser.add_argument('--tgt-parquet', default='/path/to/tgt_parquet')

    args = parser.parse_args()
    dataset = load_from_disk(args.src_hf_dataset)

    # add a row to each data item that represents a unique id
    def make_map_fn():

        def process_fn(example, idx):
            problem = example.pop('question')
            answer = example.pop('answer')
            images = example.pop('image')


            num_img_tags = problem.count("<image>")
            # some problems with only 1 image does not contain the <image> tag
            if num_img_tags != len(images):
                problem = "<image>" + problem
            data = {
                "data_source": 'virl39k',
                "prompt": [
                    {"role": "system", "content": 'You are given an image and a relevant question. Based on the query, please describe the image in detail. Do not try to answer the question.'},
                    {"role": "user", "content": f'Question: {problem}' + "\n" + "Please describe the image. DO NOT try to answer the question!"}
                ],
                "images": images,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': 'train',
                    'index': idx,
                    'answer': answer,
                    "question": problem,
                    'category': example['category'],
                    'source': example['source'],
                    'qid': example['qid'],
                }
            }
            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn(), with_indices=True, num_proc=8)
    dataset.to_parquet(args.tgt_parquet)