# GPT Business Complexity model

The model is provided at the following Hugging Face link: [https://huggingface.co/Ties/gpt-business-complexity](https://huggingface.co/Ties/gpt-business-complexity).

See Appendix B in the [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4480309) for details on the design and training of the model. 

## Model deployment

Below  are instructions for deploying the model through Runpod.ai. The steps are similar for other deployments on Linux hardware with an NVIDIA GPU.

1. Familiarize yourself with how to use Runpod.ai pods. 
    - A good resource can be found here: [chatgpt_paper/code_examples](https://github.com/TiesdeKok/chatgpt_paper/blob/main/code_examples/runpod/fast_inference.ipynb).
2. Create a new pod on Runpod.ai with the following specifications:
    - Use a GPU with at least 16GB of VRAM.
        - In the paper we used NVIDIA H100 GPUs due to the large volume, but a smaller GPU is sufficient for smaller workloads or testing. 
        - For the demonstration below we use a pod with a single L40s GPU. 
    - Use the following container image: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
3. Connect to Jupyter or SSH into the pod and run the following commands to set up the inference environment:

```bash
pip install nvitop
pip install vllm==0.10.1.1
pip install ninja
pip install flash_attn==2.8.3 --no-build-isolation
pip install huggingface_hub
```

4. Use the following code to run inference with the model:

```python
## -------
## Imports
## -------

import os, json

from vllm import LLM, SamplingParams
import torch

from huggingface_hub import snapshot_download

## --------------
## Download model
## --------------

model_folder = "/workspace/gpt-business-complexity"

snapshot_download(
    repo_id="Ties/gpt-business-complexity",
    local_dir=model_folder
)
## ------------------
## Prompts to process
## ------------------

prompt_list = [
    '<|context|>Concurrent with the closing of the EMEA 2 Joint Venture, the EMEA 2 Joint Venture entered into credit facility agreements with a group of lenders under which it could borrow up to approximately 1.5 billion in total at the exchange rate in effect on September 30, 2021, with such facilities maturing in 2025 and 2026. In connection with our 20 equity investment in the EMEA 2 Joint Venture, we provided the lenders with guarantees covering 20 of all payments of principal and interest due and payable by the EMEA 2 Joint Venture under these credit facilities, up to a limit of 315.9 million in total at the exchange rate in effect on September 30, 2021.<|target|>As of September 30, 2021, the maximum potential amount of our future payments under these guarantees was approximately $30.4 million, at the exchange rates in effect on that date.\n####\n',
    '<|context|><|target|>The Company had contract assets of $1,664 and $1,464 , as of June 30, 2023 and December 31, 2022, respectively.\n####\n'
]

## ----------------
## Initialize model
## ----------------

sampling_params = SamplingParams(
    temperature=0, 
    logprobs = 5, 
    max_tokens = 2_000, 
    stop = ["END"]
)

llm = LLM(model=model_folder, tensor_parallel_size=torch.cuda.device_count())

## -------------
## Run inference
## -------------

outputs = llm.generate(prompt_list, sampling_params)

## --------------
## Inspect output
## --------------

for item in outputs:
    print(item.outputs[0].text)

# As of September 30, 2021, the maximum potential amount of our future payments under these guarantees was approximately $<tag standard="guarantee | obligations | maximum | exposure" custom---scheduleofequitymethodinvestmentequitymethodinvesteenameaxis="custom---emea2jointventuremember" standard---guaranteeobligationsbynatureaxis="standard---financialguaranteemember">30.4</tag> million, at the exchange rates in effect on that date.
# The Company had contract assets of $<tag standard="contract | with | customer | asset | net">1,664</tag> and $<tag standard="contract | with | customer | asset | net">1,464</tag> , as of June 30, 2023 and December 31, 2022, respectively.
```