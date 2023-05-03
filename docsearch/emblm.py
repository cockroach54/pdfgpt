import openai
from typing import Dict, List
from tqdm.auto import tqdm
from time import sleep


class OpenAiEmbLM:
    """문장간 유사도나, 근거 여부를 판단키 위한 임베딩 모델 (OPENAI API사용)"""

    def __init__(self, model_name: str = "text-embedding-ada-002", **kwargs) -> None:
        """문장간 유사도나, 근거 여부를 판단키 위한 임베딩 모델
        Args:
            model_name (str): 사용할 OpenAI 모델의 이름. 기본값은 "text-embedding-ada-002" 입니다.
        Returns:

        """
        openai.api_key = kwargs["OPENAI_API_KEY"]

        self.model_name = model_name

    def embed(self, input_string: str):
        # Creates embedding vector from user input
        embedded = openai.Embedding.create(
            input=input_string,
            model=self.model_name,
        )[
            "data"
        ][0]["embedding"]
        return embedded

    def embed_segs(self, segs: List[dict]) -> List[dict]:
        """
        Segments를 임베딩하여 리턴합니다.
        Args:
            segs (List[dict]): 임베딩할 Segments 리스트
        Returns:
            List[dict]: Segments 리스트와 embedding 정보가 추가된 리스트
        """
        batch_size = 100  # how many embeddings we create and insert at once
        embeds = []
        model_name = self.model_name
        segs = segs.copy()
        MAX_ERR_COUNT = 10

        for i in tqdm(range(0, len(segs), batch_size)):
            err_cnt = 0
            # find end of batch
            i_end = min(len(segs), i + batch_size)
            if i == i_end:
                break
            meta_batch = segs[i:i_end]
            # get texts to encode
            texts = [x["text"] for x in meta_batch]
            # create embeddings (try-except added to avoid RateLimitError)
            try:
                res = openai.Embedding.create(input=texts, engine=model_name)
            except Exception as e:
                done = False
                print(e)
                print("openai api error... retry...")
                while not done:
                    if err_cnt == MAX_ERR_COUNT:
                        raise Exception(
                            f"MAX_ERR_COUNT({MAX_ERR_COUNT}) limit occured! check your code again."
                        )
                    sleep(3)
                    try:
                        res = openai.Embedding.create(input=texts, engine=model_name)
                        done = True
                    except Exception as e:
                        err_cnt += 1
            embeds.append([record["embedding"] for record in res["data"]])

        # merge embedding to segs
        embeds_flatten = [e for b in embeds for e in b]
        len(embeds_flatten)
        assert len(segs) == len(embeds_flatten)

        for seg, v in zip(segs, embeds_flatten):
            seg["embedding"] = v

        return segs


class LocalEmbLM:
    """문장간 유사도나, 근거 여부를 판단키 위한 임베딩 모델"""

    def __init__(
        self, model_name: str = "kogpt", model_injection=None, **kwargs
    ) -> None:
        """문장간 유사도나, 근거 여부를 판단키 위한 임베딩 모델
        Args:
            model_name (str): 사용할 모델의 이름. 기본값은 "kogpt" 입니다.
        Returns:

        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.device = "mps"
        self.model_max_len = 2048
        # self.model_max_len = 1024

        if self.model_name == "kogpt":
            DEVICE = self.device
            tokenizer = AutoTokenizer.from_pretrained(
                "kakaobrain/kogpt",
                revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
                bos_token="[BOS]",
                eos_token="[EOS]",
                unk_token="[UNK]",
                pad_token="[PAD]",
                mask_token="[MASK]",
            )
            if model_injection is None:
                model = AutoModelForCausalLM.from_pretrained(
                    # './kogpt-ft-3',
                    "/Users/lsw0504/code/huggingface/ggml/examples/kogpt/models",
                    # "kakaobrain/kogpt",
                    revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
                    pad_token_id=tokenizer.pad_token_id,
                    torch_dtype=torch.float16,
                    # torch_dtype='auto',
                    low_cpu_mem_usage=True,
                ).to(DEVICE)
                _ = model.eval()
            else:
                model = model_injection
        # make embedding model (remove last fc layer)
        self.emb_lm = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.tokenizer = tokenizer

    def embed(self, input_string: str):
        """Creates embedding vector from user input (single string! not list)"""
        # Creates embedding vector from user input
        from .utils import embed

        embedded = embed(
            [input_string], self.emb_lm, self.tokenizer, self.device, self.model_max_len
        )
        return embedded[0]

    def embed_segs(self, segs: List[dict]) -> List[dict]:
        """
        Segments를 임베딩하여 리턴합니다.
        Args:
            segs (List[dict]): 임베딩할 Segments 리스트
        Returns:
            List[dict]: Segments 리스트와 embedding 정보가 추가된 리스트
        """
        from .utils import embed

        batch_size = 1  # how many embeddings we create and insert at once
        embeds = []
        for i in tqdm(range(0, len(segs), batch_size)):
            # find end of batch
            i_end = min(len(segs), i + batch_size)
            if i == i_end:
                break
            meta_batch = segs[i:i_end]
            # get texts to encode
            texts = [x["text"] for x in meta_batch]
            # create embeddings
            emb = embed(
                texts, self.emb_lm, self.tokenizer, self.device, self.model_max_len
            )
            embeds.append(emb)

        # merge embedding to segs
        embeds_flatten = [e for b in embeds for e in b]
        len(embeds_flatten)
        assert len(segs) == len(embeds_flatten)

        for seg, v in zip(segs, embeds_flatten):
            seg["embedding"] = v

        return segs
