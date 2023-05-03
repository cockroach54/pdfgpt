import os
import json
from pprint import pprint
from typing import Dict, List

from .lm import LocalLM, OpenAiLM
from .emblm import LocalEmbLM, OpenAiEmbLM
from .vectordb import RediSearch, JsonSearch
from .segment import DocAssistant

realpath = os.path.dirname(os.path.realpath(__file__))


class DocGPT:
    """문서내 질문을 수행하는 언어 모델의 메인 객체"""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        emb_model_name: str = "text-embedding-ada-002",
        vector_db: str = "redisearch",
        config: Dict = None,
    ) -> None:
        """문서내 질문을 수행하는 언어 모델의 메인 객체
        Args:

        Returns:

        """
        self.model_name = model_name
        self.emb_model_name = emb_model_name
        self.vector_db = vector_db

        # set query LM
        if self.model_name == "gpt-3.5-turbo":
            self.lm = OpenAiLM(
                model_name=self.model_name, OPENAI_API_KEY=config["OPENAI_API_KEY"]
            )
        elif self.model_name == "kogpt":
            self.lm = LocalLM(
                model_name=self.model_name, OPENAI_API_KEY=config["OPENAI_API_KEY"]
            )
        else:
            raise Exception(
                f'model_name({self.model_name}) is not supported!. must be one of ["text-embedding-ada-002", "kogpt"]'
            )

        # set embedding LM
        if self.emb_model_name == "text-embedding-ada-002":
            self.embLM = OpenAiEmbLM(
                model_name=self.emb_model_name, OPENAI_API_KEY=config["OPENAI_API_KEY"]
            )
            self.emb_dim = 1536  # text-embedding-ada-002
        elif self.emb_model_name == "kogpt":
            if self.model_name == self.emb_model_name:  # vram 절약위해 모델 재사용
                self.embLM = LocalEmbLM(
                    model_name=self.emb_model_name, model_injection=self.lm.model
                )
            else:
                self.embLM = LocalEmbLM(
                    model_name=self.emb_model_name,
                )
            self.emb_dim = 4096  # kogpt
        else:
            raise Exception(
                f'emb_model_name({self.emb_model_name}) is not supported!. must be one of ["text-embedding-ada-002", "kogpt"]'
            )

        # set vector database
        if self.vector_db == "redisearch":
            self.db = RediSearch(
                emb_dim=self.emb_dim,
                host=config.get("REDIS_HOST"),
                port=config.get("REDIS_PORT"),
                password=config.get("REDIS_PASSWORD"),
            )
        elif self.vector_db == "json":
            self.db = JsonSearch()
        else:
            raise Exception(
                f'vector_db({self.vector_db}) is not supported!. must be one of ["redisearch", "json"]'
            )

        self.assistant = DocAssistant()

    def register(self, fpath: str, save_json: bool = True, use_backup=True):
        """질문을 위한 문서 등록
        Args:
        - fpath (str): 문서 파일 경로
        - save_json (bool): 임베딩을 저장할 json 파일 생성 여부 (default: True)
        - use_backup (bool): 기존 json 파일을 사용할지 여부 (default: True)

        Returns:
        - None
        """
        if self.emb_model_name == "text-embedding-ada-002":
            dbdir = "db-ada-002"
        elif self.emb_model_name == "kogpt":
            dbdir = "db-kogpt"
        else:
            raise Exception(
                f'emb_model_name({self.emb_model_name}) is not supported!. must be one of ["text-embedding-ada-002", "kogpt"]'
            )

        fname = os.path.basename(fpath)
        opath = os.path.abspath(
            os.path.join(realpath, f"../{dbdir}/{os.path.splitext(fname)[0]}.json")
        )
        is_backup_loaded = False

        if os.path.exists(opath) and use_backup:
            with open(opath, "r") as f:
                segs = json.load(f)
                print("backup embededing json file is found. use it now!")
                is_backup_loaded = True
        else:
            # read and make segments
            segs = self.assistant.segment_topic(fpath)
            # embed segments
            segs = self.embLM.embed_segs(segs)

        if self.vector_db == "redisearch":
            # reset all keys (임시로!!!)
            self.db.redis_client.flushall()
            # register segments to db index
            self.db.index_documents(segs)
        elif self.vector_db == "json":
            # register segments to db index
            self.db.index_documents(segs)

        # save as json backup
        # store embeddigs as files(db)
        if save_json and not is_backup_loaded:
            with open(opath, "w") as f:
                json.dump(segs, f, ensure_ascii=False, indent=4)
            print(f"embedding json saved at {opath} for backup.")

    def query(self, nl_query: str, topn: int = 3):
        """자연어로 문서내 내용 질문하기
        Args:
        - nl_query (str): 자연어 질문
        - topn (int): 검색된 세그먼트 중 반환할 세그먼트 개수 (default: 3)

        Returns:
        - ans_list (list): 각 질문에 대한 답변 리스트
        """
        # make embedding from nl query
        embedded_query = self.embLM.embed(nl_query)

        # search segs
        print("-- run sementic search ---")
        segments = self.db.sementic_search(embedded_query, k=topn)
        print("-- run sementic search end ---")
        print()

        # query & return
        ans_list = []
        for i, ctx in enumerate(segments.docs):
            ans = self.lm.query(nl_query, ctx)
            ans_list.append(ans)
            print(f"[Query]: {nl_query}")
            print(f"[Answer{i+1}]:")
            print(ans)
            print("-" * 10)

        return ans_list
