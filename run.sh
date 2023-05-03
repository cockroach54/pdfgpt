### change working directory to the directory of this script
cd $(dirname "$0")

# python main.py --task register --fpath "./source/[GIP] 품목별 ICT 시장동향_인공지능_2022.pdf"
# python main.py --task register --fpath "./source/sds-chatgpt.pdf"

python main.py --task all\
    --fpath "./source/[GIP] 품목별 ICT 시장동향_인공지능_2022.pdf"\
    --query "현재 인공지능 시장을 선두하고 있는 국가는 어디인가?"

# python main.py --task all\
#     --fpath "./source/2023년+미국콘텐츠산업동향+3호.pdf"\
#     --query "z세대 이외에 세대를 구분하는 기준은 무엇이 있나요?"

# python main.py --task all\
#     --fpath "./source/sds-chatgpt.pdf"\
#     --query "gpt3.5와 gpt4는 어떤 관계인가?"

# python main.py --task all\
#     --fpath "./source/sds-chatgpt.pdf"\
#     --query "gpt3.5에 비해 gpt4는 어떤 부분이 개선되었나?"

# python main.py --task all\
#     --fpath "./source/2023년 7대 국내 트렌드.pdf"\
#     --query "2023년도를 선도할 가장 중요한 트렌드 키워드는 무엇인가?"
