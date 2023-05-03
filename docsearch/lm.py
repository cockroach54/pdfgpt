import openai

OPENAI_PROMPT_TEMPLATE = """\
주어진 본문을 바탕으로 아래 질문에 대해 대답해줘.
\n### 본문
{}
\n### 질문
{}
"""

# for kogpt
LOCAL_PROMPT_TEMPLATE = """\
Below is an instruction that describes a task, paired with an input that provides further context.

Write a detailed response that appropriately completes the request.
### Instruction:
{}

### Input:
{}

### Response:
"""


class OpenAiLM:
    """근거 문단과 자연어 쿼리를 받아 답변을 생성하는 메인 언어모델"""

    def __init__(self, model_name: str = "gpt-3.5-turbo", **kwargs) -> None:
        """
        Args:
            model_name (str): 사용할 GPT-3 모델 이름. 기본값은 "gpt-3.5-turbo" 입니다.

        Returns:
            None
        """
        openai.api_key = kwargs["OPENAI_API_KEY"]

        self.model_name = model_name
        self.prompt_template = OPENAI_PROMPT_TEMPLATE

    def _render_prompt(self, query: str, context: str):
        """
        자연어 쿼리와 근거 문단을 이용하여 프롬프트를 생성합니다.

        Args:
            query (str): 자연어 쿼리
            context (str): 근거 문단

        Returns:
            str: 생성된 프롬프트 문자열
        """
        prompt = self.prompt_template.format(context, query)
        return prompt

    def query(self, query: str, ctx: dict):
        """
        자연어 쿼리와 근거 문단 정보를 이용하여 답변을 생성합니다.

        Args:
            query (str): 자연어 쿼리
            ctx (dict): 근거 문단 정보를 담은 딕셔너리

        Returns:
            str: 생성된 답변과 근거 정보가 포함된 문자열
        """
        # render prompt
        prompt = self._render_prompt(query, ctx["text"])
        # print(prompt)

        ### input prompot using open chatGPT api
        history = [{"role": "system", "content": "You are a helpful assistant."}]
        history.append({"role": "user", "content": prompt})

        # # print('[len history]:', len(history))  # for debug
        # pprint(prompt)
        response = openai.ChatCompletion.create(model=self.model_name, messages=history)
        # pprint(response)

        ans = response["choices"][0]["message"]["content"]
        ans_with_ref = f"""\
{ans}

[본 답변에 대한 근거로 "{ctx['source']}"의 {int(ctx['start_page'])+1}페이지의 {int(ctx['start_sidx'])+1}번째 문장부터 {int(ctx['end_page'])+1}페이지의 {int(ctx['end_sidx'])+1}번째 문장까지를 참조하였습니다. ({ctx['text'][:10]} ... {ctx['text'][-10:]})]\
"""
        return ans_with_ref


class LocalLM:
    """근거 문단과 자연어 쿼리를 받아 답변을 생성하는 메인 언어모델"""

    def __init__(
        self, model_name: str = "kogpt", model_injection=None, **kwargs
    ) -> None:
        """
        Args:
            model_name (str): 사용할 GPT-3 모델 이름. 기본값은 "kogpt" 입니다.

        Returns:
            None
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.device = "mps"
        self.prompt_template = LOCAL_PROMPT_TEMPLATE
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
        self.tokenizer = tokenizer
        self.model = model

    def _render_prompt(self, query: str, context: str):
        """
        자연어 쿼리와 근거 문단을 이용하여 프롬프트를 생성합니다.

        Args:
            query (str): 자연어 쿼리
            context (str): 근거 문단

        Returns:
            str: 생성된 프롬프트 문자열
        """
        prompt = self.prompt_template.format(context, query)
        return prompt

    def query(self, query: str, ctx: dict):
        """
        자연어 쿼리와 근거 문단 정보를 이용하여 답변을 생성합니다.

        Args:
            query (str): 자연어 쿼리
            ctx (dict): 근거 문단 정보를 담은 딕셔너리

        Returns:
            str: 생성된 답변과 근거 정보가 포함된 문자열
        """
        tokenizer = self.tokenizer
        model = self.model
        DEVICE = self.device
        # render prompt
        prompt = self._render_prompt(query, ctx["text"])
        # print(prompt)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        gen_tokens = model.generate(
            input_ids,
            # max_new_tokens=1024,
            num_return_sequences=1,
            temperature=0.8,
            no_repeat_ngram_size=6,
            do_sample=True,
        )
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        st_idx = gen_text.find("### Response:") + len("### Response:")

        ans = gen_text[st_idx:]
        ans_with_ref = f"""\
{ans}

[본 답변에 대한 근거로 "{ctx['source']}"의 {int(ctx['start_page'])+1}페이지의 {int(ctx['start_sidx'])+1}번째 문장부터 {int(ctx['end_page'])+1}페이지의 {int(ctx['end_sidx'])+1}번째 문장까지를 참조하였습니다. ({ctx['text'][:10]} ... {ctx['text'][-10:]})]\
"""
        return ans_with_ref
