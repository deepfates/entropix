# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import torch
from torch import Tensor
from entropix.config import ModelParams
from entropix.torch_device import get_device
from entropix.torch_kvcache import KVCache
from entropix.torch_model import xfmr
from entropix.torch_weights import XfmrWeights
from entropix.torch_main import build_attn_mask, precompute_freqs_cis, sample
from entropix.tokenizer import Tokenizer
from entropix.torch_weights import load_weights
from entropix.config import LLAMA_1B_PARAMS
DEFAULT_INSTRUCT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = get_device()
        self.tokenizer = Tokenizer('entropix/tokenizer.model')
        self.model_params = LLAMA_1B_PARAMS
        self.xfmr_weights = load_weights()

    def generate(self,
        xfmr_weights: XfmrWeights,
        model_params: ModelParams,
        tokens: Tensor,
    ) -> str:
        """Run a single prediction on the model"""
        gen_tokens = None
        cur_pos = 0
        tokens = torch.tensor([tokens], dtype=torch.long).to(self.device)
        bsz, seqlen = tokens.shape
        attn_mask = build_attn_mask(seqlen, cur_pos)
        freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
        kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim).to(self.device)
        logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
        next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
        gen_tokens = next_token
        output = self.tokenizer.decode([next_token.item()])
        cur_pos = seqlen
        stop = torch.tensor([128001, 128008, 128009], device=self.device, dtype=torch.int32)
        while cur_pos < 8192:
            cur_pos += 1
            logits, kvcache, scores, stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
            next_token = sample(gen_tokens, logits, scores)
            gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
            output += self.tokenizer.decode(next_token.tolist()[0])
            if torch.isin(next_token, stop).any():
                break
        return output
    
    def format_prompt(self, prompt: str, prompt_template: str) -> str:
        formatted_prompt = prompt_template.format(prompt=prompt)
        return formatted_prompt

    def predict(
        self,
        prompt: str = Input(description="Prompt to generate from"),
        prompt_template: str = Input(description="Prompt template to use", default=DEFAULT_INSTRUCT_TEMPLATE),
    ) -> str:
        """Generate a response from the model"""
        formatted_prompt = self.format_prompt(prompt, prompt_template)
        raw_tokens1 = self.tokenizer.encode(formatted_prompt,  bos=False, eos=False, allowed_special='all')
        return self.generate(self.xfmr_weights, self.model_params, raw_tokens1)
