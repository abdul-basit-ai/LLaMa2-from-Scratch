from typing import Optional, List
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer

class LLaMa:
    
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.model_args  = ModelArgs
        
    @staticmethod
    def build(checkpoints_dir : str, tokenizer_path : str, load_model: bool, max_seq_length : int, max_batch_size, device : str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob('*.pth'))
            assert len(checkpoints)>0, "No checkpoint file found"
            check_path = checkpoints[0]
            print("Loading Checkpoints", check_path)
            checkpoints = torch.load(check_path, map_location="cpu")
            print("Loading Time", time.time()-prev_time)
            prev_time = time.time()
            
        with open(Path(checkpoints_dir)/ "params.json","r") as f:
            params = json.load(f.read())
            
        model_args : ModelArgs = ModelArgs(
            max_seq_len=max_seq_length,
            max_batch_size=max_batch_size,
            device=device,
            **params)
        
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()
        
        if device == "cpu":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
            
        model = Transformer(model.args).to(device)
        
        if load_model:
            del checkpoints["rope.freqs"]
            model.load_state_dict(checkpoints, strict=True)
            print("TimeTaken to Load Model", time.time()-prev_time)
        
        return LLaMa(model, tokenizer, model_args)
    
    def text_completion(self, prompt: list[str], temperature: float=0.6, top_p:float=0.9, max_gen_len : Optional[int]=None):
        if max_gen_len is None:
            max_gen_len=self.args.max_seq_len
            #Prompt to Token
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos = True, add_eos = False) for prompt in prompts]
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_gen_len <= self.args.max_seq_len
        
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len )
        
        #List that will havee the generateed tokens with propmpts
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        
        for k , t in enumerate(prompt_tokens):
            tokens[k,:len(t)]= torch.tensor(t, dtype=torch.long, device=device)
            
            eos_reached = torch.tensor([False] * batch_size, device=device)
            prompt_tokens_mask = tokens != pad_id
            
        for cur_pos in tqdm(range(1,total_len), desc="Generating Tokens"):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:,-1]/temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:,-1], dim=-1)
                
                
            next_token = next_token.reshape(-1)
            #Replace tkn if its a pad token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos],tokens[:,cur_pos], next_token)
            tokens[:,cur_pos]=next_token
            #EOS is reached only if we find in padding pos
            eos_reached |= (~prompt_tokens_mask[:,cur_pos]) & (next_token==self.tokenizer.eos_id())
            if all(eos_reached):
                break
            
            out_tokens = []
            out_text = []
            for prompt_index , current_prompt_tokens in enumerate(tokens.tolist()):
                if self.tokenizer.eos_id() in current_prompt_tokens:
                    eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                    current_prompt_tokens = current_prompt_tokens[:eos_idx]
                
                out_tokens.append(current_prompt_tokens)
                out_text.append(self.tokenizer.Decode(current_prompt_tokens))
            return (out_tokens,out_text)
            
    def _sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort,dim=-1)
        mask = probs_sum - probs_sort >p #shifting
        probs_sort[mask] = 0.0    # 0 other we not selecyted
        #Re distribute to make t sum ==1
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort,num_samples=1)
        next_token = torch.gather(probs_idx,-1,next_token)
        return next_token
    
if __name__ == 'main':
    torch.manual_seed(0)
    allow_cuda = False
    device = "cuda" if torch.cuda.is_available()  and allow_cuda else "cpu"
    
    prompts = [
        "Simply Put,the newtons third law states that",
        """Tell me if the following person is actually Doremon disguised as Human:
        Name : ABdul Basit
        Decision: """
    ]
    
    model = LLaMa.build(
        checkpoints_dir='llama-2-7b/',
        load_model=True,
        max_seq_length=1024,
        max_batch_size=len(prompts),
        device=device
    )
    
    print("Successful")
    
    #Inference 
    out_tokens, out_text = (model.text_completion(prompts, max_gen_len = 64 ))
    assert len(out_text) == len(prompts)
    for i in range (len(out_text)):
        print(f'{out_text[i]}')
        print('-',50)