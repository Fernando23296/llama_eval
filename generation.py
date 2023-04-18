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

'''
ESTE CODIGO FUNCIONA PARA GENERAR PALABRAS LIMITADAS
O SEA SI LE INDUCES A QUE DIGA ABERDEEN Y ABERDEEN ESTA EN LAS POSIBLES RESPUESTAS ENTONCES FUNCIONARA BIEN
SINO HARA UN LOOP ETERNO PORQUE LAS OPCIONES SON DISTINTAS Y NO MANEJA UNA MEMORIA PARA VOLVER A HACER DE CERO.

'''



#goal = [20626, 311, 264]
#matrix = [[98, 21, 456], [20626, 311, 264, 99], [20626, 311], [20626, 311, 264]]
#matrix = [[5459, 27014, 983], [14325], [10557], [4092], [2163, 5070], [438, 346, 4807]]
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
        relation: str = 0
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
        
        #possible_list = ["Antarctica", "Aberdeen"]
        
        #print("total_len:", total_len)
        #print("start_pos:", start_pos)
        #print("TOKENS:")
        #print(tokens)
        original_tokens = [nt for nt in tokens.tolist()[0] if nt != -1]
        #print("/"*20)
        #print("SUBTOKENS:")
        #print(decode_aux(original_tokens))
        #print("_"*20)
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
            #print("TOKENS!"*10)
            #print(len(tokens_))
            #print(tokens_)
            
            is_first_token = True
            tracking_probs = []
            
            flag_word_of_interest = True
            
      
            for cur_pos in range(start_pos, total_len):
                #repeated_token = True
                logits = self.model.forward(tokens_[:, prev_pos_:cur_pos], prev_pos_)
                if temperature > 0:
                    cc = 0
                    probs = torch.softmax(logits / temperature, dim=-1)
                    #print("PROBS GENERATIO:", probs)
                    next_token, prob_next_token, flag = sample_top_p(probs, top_p, tracking_tokens)
                    
                    #print("TOKEN RECIBIDO 1")
                    #print(next_token)
                    probs_tokens_multiplicated = probs_tokens_multiplicated * prob_next_token
                    #print("FLAAAG 2")
                    #print(probs_tokens_multiplicated)
		                 
                #else:
                #next_token = torch.argmax(logits, dim=-1)
                
                
                #print("TOKEN RECIBIDO 2")
                #print(next_token)
                next_token = next_token.reshape(-1)
                next_token = torch.where(
                input_text_mask[:, cur_pos], tokens_[:, cur_pos], next_token)
                tokens_[:, cur_pos] = next_token
                #print("/+"*20)
                #print(next_token)
                filter_tokens = [nt for nt in tokens_.tolist()[0] if nt != -1]
                #filter_tokens_2 = []
                possible = decode_aux(filter_tokens)
                print("Sequence so far:")
                print(possible)
                #possible = possible.split(" ")
                #print("POSSIBLE:", possible)
                if prob_next_token == -1:
                    flag_while = False
                    
                    
                if flag == "END":
                    #print("TEEEEEEEEEEEEEEERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRMMMMMMMMMMMMMMMMMMMMMIIIIIIIIIIIIIIIIINAAAAAAAAAAAAAAAAAAAARRRRRR")
                    flag_while = False
                    #break_while = False
                    break
                #break
                '''
                for ft in filter_tokens[len(original_tokens):]:
                        
                    if ft == 29889:
                        flag_point = True
                        break
                    else:
                        filter_tokens_2.append(ft)
		        
		        
                    if len(filter_tokens_2) > 1:
                        flag_point = True
			
                    prev_pos_ = cur_pos
                    print("TRACKING TOKENS:")
                    print(tracking_tokens)
                        
                    if flag_point:
                        #print(filter_tokens_2)
                        possible = decode_aux(filter_tokens_2)
                        possible = possible.split(" ")
                        print("POSSIBLE:", possible)
                            
                        if possible[0] in possible_list:
                            print("YES")
                            print(tracking_tokens)
                            print(tracking_probs)
                            break_while = True
                            break
                    #if_first_token = False
               '''
        #print("0"*20)
        #print(tokens_)
        #print("0"*20)
        #print("probs_tokens_multiplicated:", probs_tokens_multiplicated)
        #print("TOKENS LIST")
        #print("OTRO:", aux_tokens_)
        print(tokens_.tolist())
	# HERE I DELETE THE -1 THAT "SOBRAN"
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
        
        #print("#"*20)
        #print("Decoded")
        #print(decoded)
        #print("#"*20)
        print("PROBABILITY OF NEXT WORD", probs_tokens_multiplicated)
        return decoded

helper_list = []
flag_completo = True
matrix_helper = matrix
def sample_top_p(probs, p, tracking_tokens):

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    #next_token = torch.multinomial(probs_sort, num_samples=1)
    #print("PROBS SORT 2:",probs_sort)
    #print("LEN PROBS SORT 2:", len(probs_sort[0]))
    #print(probs_sort.item())
    next_token = torch.multinomial(probs_sort, num_samples=len(probs_sort[0]))
    #next_token = torch.gather(probs_idx, -1, next_token)
    #print("NEXT TOKEN:")
    #print(next_token)
    count_probs = 0
    
    global flag_completo
    global helper_list
    # may be not global later
    global matrix
    global matrix_helper
    #global matrix_helper
    for nt in next_token[0]:
    
        # todo lo de abajo
        
        #print("NTTT:", nt)  
        next_token = torch.gather(probs_idx[0], -1, nt)
        ntk = sp_model.decode(next_token.item())
        #print(probs[next_token.item()], next_token.item())
        #print("POSSSSSSSSSSSSSSSSSSIBLE:", str(ntk))
        global aux_index
        #probs_next_token = probs_sort[count_probs]
        probs_list_ = probs.tolist()[0]
        #print("LEN PROBS", len(probs_list_))
        probs_next_token = probs_list_[next_token.item()]
        #if len(goal) - aux_index > 0:
        flag = "OK"
        
        
        
        column_of_interest = [i[0] for i in matrix]

        if next_token.item() in column_of_interest:
            print("NEXT TOKEN")
            print(next_token.item())
            helper_list.append(next_token.item())
            
            matrix = [fila for fila in matrix if fila[0] == next_token.item()]
            matrix = [fila[1:] for fila in matrix]
            print("NEXT TOKEN PROBABILITY")
            print(probs_list_[next_token.item()])

            
            #print("HELPER LIST")
            #print(helper_list)
            
            #print("LEN MATRIX", len(matrix))
            
            #print("matrix")
            #print(matrix)
            if helper_list in matrix_helper:
                 flag = "END"
                 #print("DOOOOOOOOOOOOOONNNNNNNNNNE")
                 break
            
            break
        
        
        
    #print("NEXTTTT TOKEN:", next_token)
    #print("NEXTTTT PROBS TOKEN:", probs_next_token)
    print("_"*20)
    return next_token, probs_next_token, flag

