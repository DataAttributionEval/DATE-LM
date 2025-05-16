import argparse
import requests
import json
from tqdm import tqdm

def read_jsonl(file_path):
    """
    Reads a JSONL file and returns a list of Python dicts.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_object = json.loads(line.strip())
            data.append(json_object)
    return data

def main():
    parser = argparse.ArgumentParser(description="Run OpenAI Moderation API on a JSONL dataset.")
    parser.add_argument("--api_key", type=str, required=True,
                        help="OpenAI API key.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output JSON file where results will be saved.")

    args = parser.parse_args()

    # Set up your request parameters
    api_key = args.api_key
    input_file = args.input_file
    output_file = args.output_file

    url = "https://api.openai.com/v1/moderations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = read_jsonl(input_file)

    results = []
    for i in tqdm(range(len(data)), desc="Processing"):
        content = data[i]['messages'][0]['content'] + data[i]['messages'][1]['content']
        data_api = {
            "input": content
        }

        response = requests.post(url, headers=headers, json=data_api)
        try:
            results.append(response.json()['results'])
        except Exception as e:
            print(f"Warning: unable to parse response for item {i} -> {e}")
            results.append(None)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
