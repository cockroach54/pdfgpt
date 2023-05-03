from typing import List, Tuple
import PyPDF2
from pprint import pprint
import os
from tqdm.auto import tqdm


class DocAssistant:
    """토픽 세그먼트화, 새로운 뉴스 데이터 제공 등 쿼리에 필요한 근거 문단을 만들어주는 조력자"""

    def __init__(self) -> None:
        """토픽 세그먼트화, 새로운 뉴스 데이터 제공 등 쿼리에 필요한 근거 문단을 만들어주는 조력자
        Args:

        Returns:

        """
        pass

    def _make_setences_list(self, fpath: str) -> Tuple[List[Tuple[str, int, int]], str]:
        """문서를 읽어서 문장별로 나눈 자료구조 생성
        [("...", page, sidx), ("...", page, sidx), ...]
        """
        ext = os.path.splitext(fpath)[-1]
        fname = os.path.basename(fpath)
        assert ext in [".pdf"], f"({ext}) extention file is not allowed now..."

        # get text
        raw_data = []
        with open(fpath, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            # 모든 페이지에서 텍스트 추출
            for i, page in enumerate(pdf_reader.pages):
                raw_data.append((page.extract_text(), i))

        # refine text
        # [("...\n ... \n ...", page), ...]
        data = [
            ([(d, i) for i, d in enumerate(x[0].split("\n"))], x[1]) for x in raw_data
        ]
        # [[[("...", sidx), ("...", sidx), ...], page], ...]
        data = [
            {"text": y[0].strip(), "page": x[1], "sidx": y[1]}
            for x in data
            for y in x[0]
        ]  # flatten
        # [("...", page, sidx), ("...", page, sidx), ...]
        data = list(filter(lambda x: len(x["text"]) > 0, data))  # sanitizing
        # pprint(data)
        return data, fname

    def segment_topic(self, fpath: str, method: str = "window", **kwargs) -> List[dict]:
        """입력한 도큐먼트를 topic별로 segment화"""
        ### make knowedge base from doc files
        data, fname = self._make_setences_list(fpath)

        assert method in ["window"], f"({method}) is not allowed mothod now..."

        if method == "window":
            # segment 나누기
            segs = []
            window = kwargs.get("window", 40)  # number of sentences to combine
            stride = kwargs.get(
                "stride", 4
            )  # number of sentences to 'stride' over, used to create overlap

            i = 0  # tqdm과 enumerate 같이쓰면 버그있어서 이렇게 직접 카운터 선언
            for segidx in tqdm(range(0, len(data), stride)):
                i_end = min(len(data) - 1, i + window)
                if i == i_end:
                    break
                # print(i, i_end)
                text = " ".join([x["text"] for x in data[i:i_end]])
                # create the new merged dataset
                segs.append(
                    {
                        "source": fname,
                        "segidx": segidx,
                        "start": {"page": data[i]["page"], "sidx": data[i]["sidx"]},
                        "end": {
                            "page": data[i_end]["page"],
                            "sidx": data[i_end]["sidx"],
                        },
                        "text": text,
                        "embedding": None,
                    }
                )
                i += 1
        else:
            pass
        # pprint(segs)
        return segs

    def search_news(self, keywords: List[str]):
        """검색 api를 이용해 뉴스에서 참조할 도큐먼트를 가져옴"""
        pass
