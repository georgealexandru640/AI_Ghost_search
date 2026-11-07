                                                                            


import base64                                   
import hashlib 
import json  
import math  
from collections import Counter  
from dataclasses import asdict, dataclass 
from pathlib import Path                                          

import torch                                             
from torch import nn                                                 
from torch.nn import functional as F                                            


SPECIAL_TOKENS = {  
    "<|bos|>": 256,  
    "<|eos|>": 257, 
    "<|system|>": 258,  
    "<|user|>": 259,  
    "<|assistant|>": 260, 
    "<|tool|>": 261,  
}  
FIRST_MERGE_ID = 262                                       


def _merge_pair(sequence, pair, new_id):  
    merged = []  
    index = 0  
    while index < len(sequence):  
        if index + 1 < len(sequence) and tuple(sequence[index : index + 2]) == pair:  
            merged.append(new_id) 
            index += 2  
        else:  
            merged.append(sequence[index])  
            index += 1  
    return merged  

class GhostBPETokenizer: 
    def __init__(self):                                           
        self.special_tokens = dict(SPECIAL_TOKENS) 
        self.vocab = {byte: bytes([byte]) for byte in range(256)}  
        self.merges = []                                          

    @property                                                       
    def vocab_size(self):                                                
        return max(list(self.vocab) + list(self.special_tokens.values())) + 1                             

    def train(self, texts, max_merges=128, min_frequency=2):                                   
        sequences = [list(text.encode("utf-8")) for text in texts]  
        for merge_offset in range(max_merges):  
            pair_counts = Counter( 
                pair for sequence in sequences for pair in zip(sequence, sequence[1:])                                           
            ) 
            if not pair_counts:  
                break  
            highest_frequency = max(pair_counts.values())  
            if highest_frequency < min_frequency:  
                break  
            best_pair = min(                                                                  
                pair for pair, frequency in pair_counts.items() if frequency == highest_frequency                                                 
            )  
            new_id = FIRST_MERGE_ID + merge_offset                                     
            self.merges.append((best_pair[0], best_pair[1], new_id))                             
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]                                     
            sequences = [_merge_pair(sequence, best_pair, new_id) for sequence in sequences]                                   
        return self                                  

    def encode(self, text, add_bos=False, add_eos=False, role=None):                                
        token_ids = list(text.encode("utf-8"))                               
        for left, right, new_id in self.merges:                                   
            token_ids = _merge_pair(token_ids, (left, right), new_id)                          
        prefix = []                                                  
        if add_bos:                                  
            prefix.append(self.special_tokens["<|bos|>"])                  
        if role is not None:                            
            role_token = f"<|{role}|>"                                        
            if role_token not in self.special_tokens:                           
                raise ValueError(f"Rol necunoscut: {role}")                                 
            prefix.append(self.special_tokens[role_token])                         
        if add_eos:                                  
            token_ids.append(self.special_tokens["<|eos|>"])                  
        return prefix + token_ids                                           

    def decode(self, token_ids, skip_special_tokens=True):                                       
        special_by_id = {value: key for key, value in self.special_tokens.items()}                                              
        pieces = []                                  
        byte_buffer = bytearray()                                

        def flush_bytes():                                     
            if byte_buffer:                                       
                pieces.append(bytes(byte_buffer).decode("utf-8"))                                        
                byte_buffer.clear()                           

        for token_id in token_ids:                         
            if token_id in special_by_id:                                 
                if not skip_special_tokens:                            
                    flush_bytes()                                           
                    pieces.append(special_by_id[token_id])                                    
                continue                          
            if token_id not in self.vocab:                           
                raise ValueError(f"ID de token necunoscut: {token_id}")                         
            byte_buffer.extend(self.vocab[token_id])                                             
        flush_bytes()                                       
        return "".join(pieces)                                             

    def _payload(self):                                                   
        return {                                                    
            "version": 1,                               
            "special_tokens": self.special_tokens,                              
            "merges": [                               
                {"left": left, "right": right, "new_id": new_id}                                          
                for left, right, new_id in self.merges                                 
            ],                                
            "vocab": {                                              
                str(token_id): base64.b64encode(token_bytes).decode("ascii")                                      
                for token_id, token_bytes in sorted(self.vocab.items())                                            
            },                        
        }                                    

    @staticmethod                                          
    def _hash_payload(payload):                                        
        canonical = json.dumps(                                                
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False                                                 
        ).encode("utf-8")                                
        return hashlib.sha256(canonical).hexdigest()                                    

    @property                                                        
    def fingerprint(self):                                             
        return self._hash_payload(self._payload())                                              

    def to_document(self):                                              
        payload = self._payload()                                    
        return {**payload, "sha256": self._hash_payload(payload)}                             

    @classmethod                                                      
    def from_document(cls, document):                                                
        document = dict(document)                                                  
        expected_hash = document.pop("sha256")                         
        if cls._hash_payload(document) != expected_hash:                                          
            raise ValueError("Hashul tokenizerului nu corespunde continutului.")                                         
        tokenizer = cls()                             
        tokenizer.special_tokens = {                                    
            token: int(token_id) for token, token_id in document["special_tokens"].items()                                  
        }                                           
        tokenizer.merges = [                                       
            (item["left"], item["right"], item["new_id"]) for item in document["merges"]                                       
        ]                            
        tokenizer.vocab = {                               
            int(token_id): base64.b64decode(encoded)                                                 
            for token_id, encoded in document["vocab"].items()                                         
        }                        
        return tokenizer                                     

    def save(self, path):                                             
        path = Path(path)                                                  
        path.parent.mkdir(parents=True, exist_ok=True)                                   
        path.write_text(                               
            json.dumps(self.to_document(), ensure_ascii=False, indent=2) + "\n",                                                
            encoding="utf-8",                                          
        )                                

    @classmethod                                         
    def load(cls, path):                                             
        return cls.from_document(json.loads(Path(path).read_text(encoding="utf-8")))                                                  


                                                                 
def build_causal_mask(sequence_length, device=None):                                    
    return torch.triu(                                            
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device),                                   
        diagonal=1,                                     
    )                          


                                                                
def scaled_dot_product_attention(query, key, value, mask=None):                                       
    if query.shape[-1] != key.shape[-1]:                                           
        raise ValueError("Query si key trebuie sa aiba aceeasi dimensiune.")                                  
    if key.shape[-2] != value.shape[-2]:                                                    
        raise ValueError("Key si value trebuie sa aiba aceeasi lungime.")                                  
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])                                           
    if mask is not None:                             
        scores = scores.masked_fill(mask, float("-inf"))                                      
    weights = torch.softmax(scores, dim=-1)                                                
    return weights @ value, weights                                                


class CausalAttentionHead(nn.Module):                                          
    def __init__(self, input_dim, head_dim):                                 
        super().__init__()                                    
        self.query = nn.Linear(input_dim, head_dim, bias=False)                           
        self.key = nn.Linear(input_dim, head_dim, bias=False)                         
        self.value = nn.Linear(input_dim, head_dim, bias=False)                           

    def forward(self, inputs, return_weights=False):                                      
        mask = build_causal_mask(inputs.shape[-2], device=inputs.device)                                           
        output, weights = scaled_dot_product_attention(                                    
            self.query(inputs), self.key(inputs), self.value(inputs), mask=mask                                     
        )                             
        return (output, weights) if return_weights else output                                                 


class MultiHeadCausalAttention(nn.Module):                                                     
    def __init__(self, embed_dim, num_heads=2, dropout=0.0):                                               
        super().__init__()                                    
        if embed_dim % num_heads != 0:                                               
            raise ValueError("embed_dim trebuie sa fie divizibil cu num_heads.")                                      
        self.embed_dim = embed_dim                                          
        self.num_heads = num_heads                                
        self.head_dim = embed_dim // num_heads                                
        self.heads = nn.ModuleList(                                      
            CausalAttentionHead(embed_dim, self.head_dim) for _ in range(num_heads)                             
        )                            
        self.output_projection = nn.Linear(embed_dim, embed_dim)                                   
        self.dropout = nn.Dropout(dropout)                                                   

    def forward(self, inputs, return_weights=False):                                       
        outputs, weights = zip(*(head(inputs, return_weights=True) for head in self.heads))                                                 
        projected = self.dropout(self.output_projection(torch.cat(outputs, dim=-1)))                                                   
        return (projected, torch.stack(weights, dim=1)) if return_weights else projected                                             


class GhostTransformerBlock(nn.Module):                                    
    def __init__(self, embed_dim=16, num_heads=2, mlp_ratio=4, dropout=0.0):                        
        super().__init__()                                    
        self.attention = MultiHeadCausalAttention(embed_dim, num_heads, dropout)                                         
        self.norm_after_attention = nn.LayerNorm(embed_dim)                                         
        hidden_dim = embed_dim * mlp_ratio                                            
        self.mlp = nn.Sequential(                                           
            nn.Linear(embed_dim, hidden_dim),                               
            nn.GELU(),                                                               
            nn.Linear(hidden_dim, embed_dim),                                            
            nn.Dropout(dropout),                                           
        )                   
        self.norm_after_mlp = nn.LayerNorm(embed_dim)                                   

    def forward(self, inputs, return_attention=False):                                       
        if return_attention:                                       
            attention_output, weights = self.attention(inputs, return_weights=True)                               
        else:                                
            attention_output, weights = self.attention(inputs), None                              
        hidden = self.norm_after_attention(inputs + attention_output)                                          
        output = self.norm_after_mlp(hidden + self.mlp(hidden))                                                   
        return (output, weights) if return_attention else output                                             


@dataclass(frozen=True)                                                             
class TinyGhostConfig:                                           
    vocab_size: int                                      
    context_length: int = 128                                         
    embed_dim: int = 192                                  
    num_heads: int = 4                                   
    num_layers: int = 4                                   
    mlp_ratio: int = 4                                   
    dropout: float = 0.0                                                    

    def __post_init__(self):                                          
        if self.vocab_size <= 0 or self.context_length <= 0 or self.num_layers <= 0:                                                   
            raise ValueError("Dimensiunile modelului trebuie sa fie pozitive.")                                
        if self.embed_dim % self.num_heads != 0:                                                
            raise ValueError("embed_dim trebuie sa fie divizibil cu num_heads.")                                

    def to_dict(self):                                              
        return asdict(self)                                                    

    @classmethod                                              
    def from_dict(cls, values):                                            
        return cls(**values)                                                  


class TinyGhost(nn.Module):                                         
    """Transformer autoregresiv mic: tokeni -> blocuri -> scoruri de vocabular."""                     

    def __init__(self, config):                                                       
        super().__init__()                                    
        self.config = config                                 
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)                                         
        self.position_embedding = nn.Embedding(config.context_length, config.embed_dim)                                               
        self.embedding_dropout = nn.Dropout(config.dropout)                                             
        self.blocks = nn.ModuleList(                                         
            GhostTransformerBlock(                                     
                config.embed_dim, config.num_heads, config.mlp_ratio, config.dropout                                  
            )                                 
            for _ in range(config.num_layers)                                            
        )                             
        self.final_norm = nn.LayerNorm(config.embed_dim)                                  
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)                                           

    def forward(self, token_ids, targets=None):                                       
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:                                     
            raise ValueError("token_ids trebuie sa fie tensor long cu forma (batch, secventa).")                             
        sequence_length = token_ids.shape[1]                                          
        if not 0 < sequence_length <= self.config.context_length:                                             
            raise ValueError("Lungimea secventei este invalida.")                                
        positions = torch.arange(sequence_length, device=token_ids.device)                                                 
        hidden = self.embedding_dropout(                                               
            self.token_embedding(token_ids) + self.position_embedding(positions)                                          
        )                                         
        for block in self.blocks:                                          
            hidden = block(hidden)                                     
        logits = self.lm_head(self.final_norm(hidden))                                                      
        if targets is None:                                        
            return logits                                 
        if targets.shape != token_ids.shape:                                                
            raise ValueError("targets trebuie sa aiba aceeasi forma ca token_ids.")                               
        loss = F.cross_entropy(                                               
            logits.reshape(-1, self.config.vocab_size),                                              
            targets.reshape(-1),                                                 
            ignore_index=-100,                                     
        )                              
        return logits, loss                                     


                                                                       
def count_trainable_parameters(module):                                           
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)                                              


                                              
@torch.inference_mode()                                                         
def generate_token_ids(                                                    
    model, prompt_ids, *, eos_token_id, max_new_tokens=64, temperature=0.0, generator=None                                                     
):                           
    if prompt_ids.ndim != 2 or prompt_ids.dtype != torch.long or prompt_ids.shape[1] == 0:                                                  
        raise ValueError("Promptul trebuie sa fie tensor long cu forma (batch, secventa).")                            
    if max_new_tokens < 0 or temperature < 0:                               
        raise ValueError("max_new_tokens si temperature nu pot fi negative.")                                
    was_training = model.training                                        
    model.eval()                                      
    generated = prompt_ids.clone()                                                
    for _ in range(max_new_tokens):                                               
        context = generated[:, -model.config.context_length :]                                                 
        next_logits = model(context)[:, -1, :]                                        
        if temperature == 0:                                             
            next_token = next_logits.argmax(dim=-1, keepdim=True)                                            
        else:                         
            probabilities = torch.softmax(next_logits / temperature, dim=-1)                                                   
            next_token = torch.multinomial(probabilities, 1, generator=generator)                                     
        generated = torch.cat((generated, next_token), dim=1)                                    
        if next_token.numel() == 1 and next_token.item() == eos_token_id:                                    
            break                                            
    model.train(was_training)                                          
    return generated                                            


                                                       
def generate_text(model, tokenizer, prompt, *, max_new_tokens=64, temperature=0.0, device=None):                                           
    device = device or next(model.parameters()).device                                                         
    prompt_ids = tokenizer.encode(prompt, add_bos=True, role="user")                                                     
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)                                                           
    output = generate_token_ids(                                          
        model,                    
        input_tensor,                               
        eos_token_id=tokenizer.special_tokens["<|eos|>"],                                 
        max_new_tokens=max_new_tokens,                               
        temperature=temperature,                             
    )                                
    return tokenizer.decode(output[0, len(prompt_ids) :].tolist())                                                  


def main():  
    from .pipeline import describe_role                                        

    print(describe_role("base"))                                          


if __name__ == "__main__":                                                    
    main()  