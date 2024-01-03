
import time
import sampling
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from rwkv_cpp import rwkv_cpp_shared_library, rwkv_cpp_model
from tokenizer_util import get_tokenizer

MODEL_NAME: str = "RWKV-5-World-3B-v2-20231118-ctx16k-Q5_1"
END_OF_TEXT_TOKEN: int = 0
END_OF_LINE_TOKEN: int = 187

# Model definitions
class CompletionRequest(BaseModel):
    prompt: str
    n_predict: int = 42
    temperature: float = 0.8
    top_p: float = 0.5

class GenerationSettings(BaseModel):
    model: str
    n_ctx: int = 0
    n_predict: int
    temperature: float
    top_p: float

class CompletionResponse(BaseModel):
    content: str
    generation_settings: GenerationSettings
    prompt: str
    timing: float

library = rwkv_cpp_shared_library.load_rwkv_shared_library()
model = rwkv_cpp_model.RWKVModel(library, "../models/" + MODEL_NAME + ".bin")
tokenizer_decode, tokenizer_encode = get_tokenizer('world', model.n_vocab)


def generate_completion(prompt: str, max_len_tokens: int, temperature: float, top_p: float) -> str:
    assert prompt != '', 'Prompt must not be empty'

    print('Encoding prompt')
    prompt_tokens: List[int] = tokenizer_encode(prompt)
    prompt_token_count: int = len(prompt_tokens)
    print('--- length: %i' % (prompt_token_count))

    init_logits, init_state = model.eval_sequence_in_chunks(prompt_tokens, None, None, None, use_numpy=True)


    logits, state = init_logits.copy(), init_state.copy()
    completion = ""
    accumulated_tokens: List[int] = []
    for i in range(max_len_tokens):
        token: int = sampling.sample_logits(logits, temperature, top_p)

        if (token == END_OF_TEXT_TOKEN):
            break
        accumulated_tokens += [token]

        decoded: str = tokenizer_decode(accumulated_tokens)

        print(decoded, end='', flush=True)


        logits, state = model.eval(token, state, state, logits, use_numpy=True)

        if i == max_len_tokens - 1:
            print()
            break
        
    print('generation completed', end='', flush=True)

    

    return tokenizer_decode(accumulated_tokens)

app = FastAPI()


# Endpoint
@app.post("/completion")
def completion(request: CompletionRequest) -> CompletionResponse:
    try:
        start: float = time.time()
        content = generate_completion(
            prompt=request.prompt,
            max_len_tokens=request.n_predict,
            temperature=request.temperature,
            top_p=request.top_p
        )
        delay: float = time.time() - start
        
        r = CompletionResponse()
        r.prompt = request.prompt
        r.content = content
        r.timing = delay

        gs = GenerationSettings()
        gs.model = MODEL_NAME
        gs.n_ctx = 0
        gs.n_predict = request.n_predict
        gs.temperature = request.temperature
        gs.top_p = request.top_p

        r.generation_settings = gs

        return r
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
