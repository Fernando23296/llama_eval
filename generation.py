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
    
#matrix = dict_encodes["P740"][1]
matrix_ = dict_encodes
aux_index = 0

count_helper_matrix = 0

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
        template: str = None,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        
        global count_helper_matrix
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
        
        tracking_tokens = []

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
                #repeated_token = True
                logits = self.model.forward(tokens_[:, prev_pos_:cur_pos], prev_pos_)
                if temperature > 0:
                    cc = 0
                    probs = torch.softmax(logits / temperature, dim=-1)
                    
                    next_token, prob_next_token, flag = sample_top_p(probs, top_p, tracking_tokens, template)
                   
                    probs_tokens_multiplicated = probs_tokens_multiplicated * prob_next_token
                    

                    
                next_token = next_token.reshape(-1)
                next_token = torch.where(
                input_text_mask[:, cur_pos], tokens_[:, cur_pos], next_token)
                tokens_[:, cur_pos] = next_token
                
                filter_tokens = [nt for nt in tokens_.tolist()[0] if nt != -1]
                
                
                #print("FILTER TOKENS SO FAR")
                #print(filter_tokens)
                possible = decode_aux(filter_tokens)
                #print("Sequence so far:")
                #print(possible)
                
                if prob_next_token == -1:
                    flag_while = False
                    
                    
                if flag == "END":
                    count_helper_matrix = 0
                    #print("--------------ENDING MATRIX HELPER -------------")
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
        
        slice_ = tokens_tolist_aux[0][-5:]
        d_ = decode_aux(slice_)
        d_ = d_.split(" ")
        print(probs_tokens_multiplicated, d_[-1])
        return decoded

helper_list = []
flag_completo = True
matrix_helper = 0
matrix = 0
def sample_top_p(probs, p, tracking_tokens, template):

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    
    next_token = torch.multinomial(probs_sort, num_samples=len(probs_sort[0]))
    
    count_probs = 0
    
    global flag_completo
    global helper_list
    
    global matrix
    global matrix_helper
    global count_helper_matrix
    
    if count_helper_matrix == 0:
        #print("---------------STARTING MATRIX HELPER FROM ZERO--------------")
        matrix = matrix_[template][1]
        matrix_helper = matrix_[template][1]
    
    
    for nt in next_token[0]:
    
          
        next_token = torch.gather(probs_idx[0], -1, nt)
        ntk = sp_model.decode(next_token.item())
        
        global aux_index
        
        probs_list_ = probs.tolist()[0]
        
        probs_next_token = probs_list_[next_token.item()]
        
        flag = "OK"

        
        try:
            column_of_interest = [i[0] for i in matrix]
        
        except:
            flag = "END"
            break
        


        if next_token.item() in column_of_interest:
            
            helper_list.append(next_token.item())
            
            matrix = [fila for fila in matrix if fila[0] == next_token.item()]
            matrix = [fila[1:] for fila in matrix]
            
            
            if helper_list in matrix_helper:
                 
                 
                 flag = "END"
                 #print("------ END DESDE GENERATION ------")
                 
                 
                 #nuevo
                 helper_list = []
                 break
            
            break
        
    
    count_helper_matrix += 1
    
    if next_token.item() == 29889:
        probs_next_token = 1
    return next_token, probs_next_token, flag

