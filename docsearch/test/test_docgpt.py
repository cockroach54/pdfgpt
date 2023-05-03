import os
import json
import pytest

from docgpt import DocGPT, LM, OpenAiEmbLM


@pytest.fixture(scope="module")
def docgpt():
    return DocGPT()


@pytest.fixture(scope="module")
def lm():
    return LM()


@pytest.fixture(scope="module")
def OpenAiEmbLM():
    return OpenAiEmbLM()


@pytest.fixture(scope="module")
def test_data_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "test_data"))


@pytest.fixture(scope="module")
def example_doc_path(test_data_dir):
    return os.path.join(test_data_dir, "example.txt")


@pytest.fixture(scope="module")
def example_doc_segs_path(test_data_dir):
    return os.path.join(test_data_dir, "example.json")


def test_emb_lm_embed_segs(OpenAiEmbLM, example_doc_segs_path):
    with open(example_doc_segs_path, "r") as f:
        segs = json.load(f)

    assert len(segs) == 10
    assert "embedding" not in segs[0]

    embed_segs = OpenAiEmbLM.embed_segs(segs)

    assert len(embed_segs) == 10
    assert "embedding" in embed_segs[0]


def test_lm_query(lm, example_doc_segs_path):
    with open(example_doc_segs_path, "r") as f:
        segs = json.load(f)

    query = "안녕하세요"
    ctx = segs[0]
    ans = lm.query(query, ctx)

    assert isinstance(ans, str)


def test_docgpt_register_and_query(docgpt, example_doc_path):
    docgpt.register(example_doc_path)

    nl_query = "예시 문서에서의 참조되는 텍스트는 무엇인가요?"
    ans_list = docgpt.query(nl_query)

    assert isinstance(ans_list, list)
    assert len(ans_list) == 3
    assert isinstance(ans_list[0], str)


# ------

import pytest
from docgpt import DocGPT, LM, OpenAiEmbLM, RediSearch, DocAssistant


@pytest.fixture
def docgpt():
    return DocGPT()


def test_lm_query():
    lm = LM()
    query = "문서 내에서 자연어 질문을 수행하는 클래스는 무엇인가요?"
    ctx = {"text": "DocGPT 클래스는 문서내 질문을 수행하는 언어 모델의 메인 객체입니다."}
    answer = lm.query(query, ctx)
    assert (
        answer
        == 'DocGPT 클래스는 문서내 질문을 수행하는 언어 모델의 메인 객체입니다.\n\n[본 답변에 대한 근거로 "-"의 1페이지의 0번째 문장부터 1페이지의 0번째 문장까지를 참조하였습니다. (DocGPT 클 ... 단위 테스트를 생성합니다.'
    )  # TODO: 입력 값 변경


def test_OpenAiEmbLM_embed_segs():
    OpenAiEmbLM = OpenAiEmbLM()
    segs = [
        {"text": "첫번째 문장입니다."},
        {"text": "두번째 문장입니다."},
        {"text": "세번째 문장입니다."},
    ]
    embed_segs = OpenAiEmbLM.embed_segs(segs)
    assert len(embed_segs) == 3
    assert "embedding" in embed_segs[0]
    assert "embedding" in embed_segs[1]
    assert "embedding" in embed_segs[2]
    assert len(embed_segs[0]["embedding"]) == 768  # TODO: 입력 값 변경


def test_docgpt_register_and_query(docgpt):
    fpath = "./test_file.txt"
    with open(fpath, "w") as f:
        f.write("첫번째 문장입니다.\n두번째 문장입니다.\n세번째 문장입니다.")
    docgpt.register(fpath, save_json=False)
    result = docgpt.query("첫번째 문장은 무엇인가요?")
    assert len(result) == 1
    assert "첫번째 문장입니다." in result[0]
    os.remove(fpath)  # test_file.txt 삭제
