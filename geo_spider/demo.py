import requests
import json
import time
import random
from tqdm import tqdm


def save_data(filename, data):
    with open(f'data/{filename}.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))



for page in tqdm(range(1, 100)):
    url = "https://www.geodata.cn/ManagerDev/api/geo/words/list"
    data = {
        "keywords": "",
        "contributor": "",
        "size": 100,
        "page": page,
        "orderByScore": 0,
        "orderByTime": 0
    }

    response = requests.post(url, json=data)

    save_data(str(page), response.json()['data']['list'])
    # 随机time.sleep
    sleep_time = random.randint(1, 3)  # 生成一个1到10之间的随机整数
    time.sleep(sleep_time) 
