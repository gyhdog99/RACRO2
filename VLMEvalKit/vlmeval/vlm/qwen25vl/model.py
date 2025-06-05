from __future__ import annotations

import os

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from .visual_utils import process_vision_info
from ...smp import get_rank_and_world_size
from vllm import LLM, SamplingParams
from ...smp import *


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')



class Qwen25VL(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        verbose: bool = False,
        tensor_parallel_size = 2,
        function_type = 'qa',
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
       
        self.system_prompt = system_prompt
        self.verbose = verbose
        from transformers import AutoProcessor
        assert model_path is not None
        self.model_path = model_path
        self.function_type = function_type
        
        local_rank, world_size = get_rank_and_world_size()
        if world_size > 1:
            self.llm = LLM(model=self.model_path,
                tensor_parallel_size=tensor_parallel_size,  
                gpu_memory_utilization=0.8,      
                dtype="bfloat16",
                device=f'cuda:{local_rank}',
                limit_mm_per_prompt={"image": 38},
                trust_remote_code=True,
            )
        else:
            self.llm = LLM(model=self.model_path,
                tensor_parallel_size=tensor_parallel_size,  
                gpu_memory_utilization=0.8,      
                dtype="bfloat16",
                limit_mm_per_prompt={"image": 38},
                trust_remote_code=True,
            )  
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_new_tokens,
        )
        
        self.max_new_tokens = max_new_tokens
      
  
    
    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None, function_type: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """

        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
            elif s['type'] == 'video':
                item = {'type': 'video', 'video': ensure_video_url(s['value'])}
                if self.fps is not None:
                    item['fps'] = self.fps
              
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                if function_type == 'query_cond_3':
                    item = {'type': 'text', 'text': f'Question: ' + s['value'] + "\n" + "Please describe the image. DO NOT try to answer the question!"}
                elif function_type == 'qa_cot':
                    item = {'type': 'text', 'text': s['value'].replace('Please try to answer the question with short words or phrases if possible.','')}
                else:
                    raise NotImplementedError
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

       
    def _prepare_input(self, message, dataset=None, caption=None, qa=None):
        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})

            messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset,function_type=self.function_type)})

        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        image, _ = process_vision_info([messages])

        return {
            'multi_modal_data': {'image': image},
            'prompt': text[0],
        }

