from typing import List, Dict
import redis
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.commands.search.field import (
    TextField,
    VectorField,
    NumericField,
)
from openai.embeddings_utils import cosine_similarity
import numpy as np
from abc import abstractmethod


class VectorDatabase:
    """벡터 데이터베이스 인터페이스"""

    def __init__(self) -> None:
        """벡터 데이터베이스 인터페이스
        Args:

        Returns:

        """
        pass

    @abstractmethod
    def connect(self):
        """데이터베이스에 연결"""
        pass

    @abstractmethod
    def sementic_search(self):
        """sementic serach수행
        유사도, 룰 기반, 모델 기반 등 다양한 메소드 이용 가능
        """
        pass

    @abstractmethod
    def index_documents(self, data: List[dict]):
        """데이터 등록"""
        pass


class RediSearch(VectorDatabase):
    """RediSearch 구현체 벡터 데이터베이스 인터페이스"""

    def __init__(
        self, emb_dim=int, host: str = "localhost", port: int = 6397, password: str = ""
    ) -> None:
        """벡터 데이터베이스 인터페이스
        Args:

        Returns:

        """
        self.VECTOR_DIM = emb_dim
        # self.VECTOR_DIM = len(data[0]["embedding"])  # length of the vectors
        # self.VECTOR_NUMBER = len(data)  # initial number of vectors
        self.INDEX_NAME = "gpt-idx"  # name of the search index
        self.PREFIX = "seg"  # prefix for the document keys
        self.DISTANCE_METRIC = (
            "COSINE"  # distance metric for the vectors (ex. COSINE, IP, L2)
        )

        self.redis_client = None
        # connect
        self.host = host
        self.port = port
        self.connect(self.host, self.port, password)

    def connect(self, host, port, password):
        """데이터베이스에 연결"""

        # Connect to Redis
        if self.redis_client is None:
            redis_client = redis.Redis(host=host, port=port, password=password)
            assert redis_client.ping(), "[Redis Error]: redis not connected!"
            self.redis_client = redis_client
            print("redis is newly connected :)")
        else:
            print("redis is already conneted...")

    def _check_connection(self):
        assert isinstance(
            self.redis_client, redis.Redis
        ), f"current self.redis_client is not a instance of redis.Redis but ({type(self.redis_client)})"

    def sementic_search(
        self,
        embedded_query: List[float],
        vector_field: str = "content_vector",
        return_fields: list = [
            "source",
            "text",
            "start_page",
            "start_sidx",
            "end_page",
            "end_sidx",
            "segidx",
            "vector_score",
        ],
        hybrid_fields="*",
        k: int = 3,
        print_results: bool = True,
    ) -> List[dict]:
        """sementic serach수행
        유사도, 룰 기반, 모델 기반 등 다양한 메소드 이용 가능
        """
        self._check_connection()
        redis_client = self.redis_client
        index_name = self.INDEX_NAME

        # Prepare the Query
        base_query = (
            f"{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]"
        )
        # base_query = f"KNN {k} @{vector_field} $vector AS vector_score"
        print(base_query)
        query = (
            Query(base_query)
            .return_fields(*return_fields)
            .sort_by("vector_score")
            .paging(0, k)
            .dialect(2)
        )
        params_dict = {
            "vector": np.array(embedded_query).astype(dtype=np.float32).tobytes()
        }

        # perform vector search
        results = redis_client.ft(index_name).search(query, params_dict)
        # print(results)
        if print_results:
            for i, doc in enumerate(results.docs):
                score = 1 - float(doc.vector_score)
                print(
                    f"{i}. {doc['text'][:10]} ... {doc['text'][-10:]} (Score: {round(score ,3) })"
                )
        return results

    def create_index_if_not_exist(self):
        """redisearch에 초기 인덱스 생성"""
        self._check_connection()
        redis_client = self.redis_client

        # Constants
        VECTOR_DIM = self.VECTOR_DIM  # length of the vectors
        # VECTOR_NUMBER = self.VECTOR_NUMBER  # initial number of vectors
        INDEX_NAME = self.INDEX_NAME  # name of the search index
        PREFIX = self.PREFIX  # prefix for the document keys
        DISTANCE_METRIC = (
            self.DISTANCE_METRIC
        )  # distance metric for the vectors (ex. COSINE, IP, L2)

        ### make index
        # Check if index exists
        # redis_client.ft(INDEX_NAME).dropindex()
        try:
            redis_client.ft(INDEX_NAME).info()
            print("Index already exists")
        except Exception as e:
            # Define RediSearch fields for each of the columns in the dataset
            source = TextField(name="source")
            text = TextField(name="text")
            start_page = NumericField(name="start_page")
            start_sidx = NumericField(name="start_sidx")
            end_page = NumericField(name="end_page")
            end_sidx = NumericField(name="end_sidx")
            segidx = NumericField(name="segidx")
            text_embedding = VectorField(
                "content_vector",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": VECTOR_DIM,
                    "DISTANCE_METRIC": DISTANCE_METRIC,
                    # "INITIAL_CAP": VECTOR_NUMBER,
                },
            )
            fields = [
                source,
                text,
                start_page,
                start_sidx,
                end_page,
                end_sidx,
                segidx,
                text_embedding,
            ]
            # Create RediSearch Index
            redis_client.ft(INDEX_NAME).create_index(
                fields=fields,
                definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH),
            )
            print(f"create new index {INDEX_NAME}")

    def index_documents(self, data: List[dict]):
        """redisearch 인덱스에 도큐먼트 등록"""
        self._check_connection()
        redis_client = self.redis_client
        PREFIX = self.PREFIX  # prefix for the document keys

        self.create_index_if_not_exist()

        ### load data to db index
        for seg in data:
            key = f"{PREFIX}:{str(seg['segidx'])}"

            # create byte vectors for title and content
            embedding = np.array(seg["embedding"], dtype=np.float32).tobytes()

            # converted new dict
            new_seg = {
                "source": seg["source"],
                "text": seg["text"],
                "start_page": seg["start"]["page"],
                "start_sidx": seg["start"]["sidx"],
                "end_page": seg["end"]["page"],
                "end_sidx": seg["end"]["sidx"],
                "segidx": seg["segidx"],
                "content_vector": embedding,
            }

            redis_client.hset(key, mapping=new_seg)


class JsonSearch(VectorDatabase):
    """Json 파일들을 사용하는 벡터 데이터베이스 인터페이스"""

    def __init__(self) -> None:
        """벡터 데이터베이스 인터페이스
        Args:

        Returns:

        """
        self.documents = []

    def connect(self):
        """데이터베이스에 연결"""
        print("not used method :)")

    def sementic_search(
        self,
        embedded_query: List[float],
        k: int = 3,
        print_results: bool = True,
    ) -> List[dict]:
        """sementic serach수행
        유사도, 룰 기반, 모델 기반 등 다양한 메소드 이용 가능
        """
        segs = self.documents

        # perform vector search
        sims = [
            {"idx": i, "sim": cosine_similarity(embedded_query, seg["embedding"])}
            for i, seg in enumerate(segs)
        ]
        sorted_sims = sorted(sims, key=lambda x: x["sim"], reverse=True)
        ctx_cands = [
            {
                "sim": x["sim"],
                "start_sidx": segs[x["idx"]]["start"]["sidx"],
                "end_sidx": segs[x["idx"]]["end"]["sidx"],
                "start_page": segs[x["idx"]]["start"]["page"],
                "end_page": segs[x["idx"]]["end"]["page"],
                "text": segs[x["idx"]]["text"],
                "segidx": segs[x["idx"]]["segidx"],
                "source": segs[x["idx"]]["source"],
            }
            for x in sorted_sims[:k]
        ]

        # 인터페이스 통일
        results = SearchResult(ctx_cands)

        # print(results)
        if print_results:
            for i, doc in enumerate(results):
                score = 1 - float(doc.vector_score)
                print(
                    f"{i}. {doc['text'][:10]} ... {doc['text'][-10:]} (Score: {round(score ,3) })"
                )
        return results

    def index_documents(self, data: List[dict]):
        """json search 인덱스에 도큐먼트 등록"""
        self.documents = data


from dataclasses import dataclass


@dataclass
class SearchResult:
    docs: List[Dict]
