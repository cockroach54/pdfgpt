import torch
from typing import List


def embed(text_batch: List[str], emb_lm, tokenizer, DEVICE="cuda", model_max_len=2048):
    tks_batch = [tokenizer.encode(text) for text in text_batch]

    max_token_len = max([len(x) for x in tks_batch])
    #     print(max_token_len)
    if max_token_len > model_max_len:
        print(
            f"입력한 텍스트들의 최대 토큰 길이({max_token_len})가 model_max_len({model_max_len}) 보다 크므로 슬라이싱 합니다"
        )

    padded_tks_batch = []
    for tks in tks_batch:
        if len(tks) < max_token_len:
            padded_tks = tks + [tokenizer.pad_token_id] * (max_token_len - len(tks))
        else:
            padded_tks = tks.copy()
        padded_tks_batch.append(padded_tks[:model_max_len])

    tks_tensor = torch.tensor(padded_tks_batch).to(DEVICE)
    embedding = emb_lm(tks_tensor).last_hidden_state
    del tks_tensor
    return embedding[:, -1].tolist()
