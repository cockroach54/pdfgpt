import os
import argparse
import toml
import warnings
from docsearch.docgpt import DocGPT

warnings.filterwarnings("ignore")

realpath = os.path.dirname(os.path.realpath(__file__))
# config 불러오기
with open(os.path.join(realpath, "config.toml"), "r") as f:
    cfg = toml.load(f)
    OPENAI_API_KEY = cfg["OPENAI_API_KEY"]
    SERPAPI_API_KEY = cfg["SERPAPI_API_KEY"]

# set openai api key
if OPENAI_API_KEY is None or OPENAI_API_KEY == "":
    _key = os.environ.get("OPENAI_API_KEY")
    if _key is None:
        raise Exception("OPENAI_API_KEY 를 입력하지 않았고 시스템 환경변수도 존재하지 않습니다.")
    else:
        OPENAI_API_KEY = _key

if __name__ == "__main__":
    # [Usage]: python3 train.py -t register
    ### parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        help="taks name. [register(r), query(q), all(a)], 'all' means running regist->query at once.",
        default="all",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="natural language query",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--fpath",
        "-f",
        type=str,
        help="document file path for register",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--topn",
        "-n",
        type=int,
        help="a number of query answers.",
        required=False,
        default=3,
    )

    config = parser.parse_args()

    task = config.task
    query = config.query
    fpath = config.fpath
    topn = config.topn

    # get AI instance
    docgpt = DocGPT(
        model_name="gpt-3.5-turbo",
        emb_model_name="kogpt",
        # emb_model_name="text-embedding-ada-002",
        config=cfg,
    )
    if task.lower() in ["register", "r"]:
        assert fpath is not None, "task 'register (r)' must need 'fpath' parameter!"
        docgpt.register(fpath)
        print(f'"({fpath})" is regitered normally :)')
    elif task.lower() in ["query", "q"]:
        assert query is not None, "task 'query (q)' must need 'query' parameter!"
        docgpt.query(nl_query=query, topn=topn)
    elif task.lower() in ["all", "a"]:
        assert (
            query is not None and fpath is not None
        ), "task 'all (a)' needs all parameters 'fpath', 'query'!"
        docgpt.register(fpath)
        print(f'"({fpath})" is regitered normally :)')
        docgpt.query(nl_query=query, topn=topn)

    else:
        raise Exception(f"not supportd task yet!; {task}")
