# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List
import json
import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer
from sentencepiece import SentencePieceProcessor

model_path = "/datasets/COLT/llama/tokenizer.model"
sp_model = SentencePieceProcessor(model_file=model_path)


with open('/homedtcl/bsilva/llama/new_dict_encoded.txt', 'r') as archivo:
    dict_encodes = json.load(archivo)
    
matrix = dict_encodes["P740"][1]
aux_index = 0

def decode_aux(t: List[int]) -> str:
    return sp_model.decode(t)
   
class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0

        original_tokens = [nt for nt in tokens.tolist()[0] if nt != -1]

        flag_point = False
        flag_while = True
        
        print("_"*20)
        tracking_tokens = []
        print("_"*20)
        break_while = True
        
        probs_tokens_multiplicated = 1
        
        aux_tokens_ = 0
        while flag_while:
            tokens_ = tokens.clone()
            prev_pos_ = prev_pos
            
            is_first_token = True
            tracking_probs = []
            
            flag_word_of_interest = True
            
      
            for cur_pos in range(start_pos, total_len):

                logits = self.model.forward(tokens_[:, prev_pos_:cur_pos], prev_pos_)
                if temperature > 0:
                    cc = 0
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token, prob_next_token, flag = sample_top_p(probs, top_p, tracking_tokens)                  
                    probs_tokens_multiplicated = probs_tokens_multiplicated * prob_next_token

		                 
                #else:
                next_token = torch.argmax(logits, dim=-1)
                

                next_token = next_token.reshape(-1)
                next_token = torch.where(
                input_text_mask[:, cur_pos], tokens_[:, cur_pos], next_token)
                tokens_[:, cur_pos] = next_token
                filter_tokens = [nt for nt in tokens_.tolist()[0] if nt != -1]

                possible = decode_aux(filter_tokens)
                possible = possible.split(" ")

                if prob_next_token == -1:
                    flag_while = False
                    
                    
                if flag == "END":
                    flag_while = False

                    break

        tokens_tolist_aux = [[x for x in sublist if x != -1] for sublist in tokens_.tolist()]
        decoded = []
        for i, t in enumerate(tokens_tolist_aux):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        
        return decoded

helper_list = []
flag_completo = True

def sample_top_p(probs, p, tracking_tokens):

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    next_token = torch.multinomial(probs_sort, num_samples=len(probs_sort[0]))
    count_probs = 0
    
    global flag_completo
    global helper_list
    
    for nt in next_token[0]:
        
        next_token = torch.gather(probs_idx[0], -1, nt)
        ntk = sp_model.decode(next_token.item())
        global aux_index
        probs_list_ = probs.tolist()[0]
        probs_next_token = probs_list_[next_token.item()]
        flag = "OK"

  
        helper_list.append(next_token.item())
        
       
        for secuencia in matrix:
            if len(secuencia) <= len(helper_list):
                for i in range(len(helper_list) - len(secuencia) + 1):
                    if all(secuencia[j] == helper_list[i + j] for j in range(len(secuencia))):
                        flag = "END"
                        helper_list = []
                        break
        
        
        
        break
        
    return next_token, probs_next_token, flag

