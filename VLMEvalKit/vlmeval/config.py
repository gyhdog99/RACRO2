from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial


mllm_grp = {
    'MLLM': partial(Qwen25VL, \
          model_path='path_to_ckpt',  \
          tensor_parallel_size=1, \
          max_new_tokens=8192, \
          temperature=0, \
          system_prompt = 'You are a helpful assistant.',
          function_type = 'qa_cot'
      ),
}

llm_grp = {
    'LLM': partial(Reasoner, \
          model_path='path_to_ckpt',  \
          tensor_parallel_size=1, \
          max_new_tokens=8192, \
          temperature=0.6, \
          top_p=0.95, \
          function_type="joint" \
       ),
}

supported_VLM = {}

model_groups = [
    mllm_grp,
    llm_grp
]

for grp in model_groups:
    supported_VLM.update(grp)
