import json
import pprint


with open("fake_data.json", "r", encoding='utf-8') as f:
    data = json.load(f)

res = []

for i in range(len(data)):
    system = "将文本中的name、address、email、question提取出来，以json格式输出，字段为name、address、email、question，值为文本中提取出来的内容。"
    instruction = data[i]['text']
    output = f"""```json\n{json.dumps(data[i]['structured_output'], ensure_ascii=False, indent=4)}\n```"""

    tmp = {
        "system": system,
        "instruction": instruction,
        "input": "",
        "output": output,
    }

    res.append(tmp)

with open("fake_sft.json", "w", encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=4)