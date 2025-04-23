import requests
import pprint
import json

def generate_data():
    url = "http://115.190.74.90/v1/workflows/run"
    headers = {
        "Authorization": "Bearer app-dWopgPGHnkZ77U2dPMHwQvkS",
        "Content-Type": "application/json"
    }
    data = {
        "inputs": {},
        "response_mode": "blocking",
        "user": "abc-123"
    }

    response = requests.post(url, headers=headers, json=data)

    json_response = response.json()

    return json_response['data']["outputs"]

def save_data(data):
    with open("fake_data.jsonl", "a", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    res = []
    for i in range(300):
        data = generate_data()
        res.append(data)
        save_data(data)
        print(f"Generated data {i+1}: {data}")

    with open("fake_data.json", "w", encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False)
        print(f"Saved {len(res)} data to fake_data.json")