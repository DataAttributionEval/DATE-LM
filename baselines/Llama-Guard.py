import argparse
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def read_jsonl(file_path: str):
    """
    Reads a JSONL file and returns a list of dict objects.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_object = json.loads(line.strip())
            data.append(json_object)
    return data

def moderate_batch(
    model,
    tokenizer,
    conversations,
    unsafe_token_id: int,
    safe_token_id: int,
    batch_size: int = 8,
    device: str = "cuda"
):
    """
    Perform batched generation and extract "unsafe" probabilities at the third token.
    :param model: A causal LM loaded with transformers
    :param tokenizer: The corresponding tokenizer
    :param conversations: List of conversation data (each item is 'messages')
    :param unsafe_token_id: ID of the "unsafe" token
    :param batch_size: Number of conversations per inference batch
    :param device: 'cuda' or 'cpu'
    :return: List of dicts: [{"moderation": str, "unsafe_prob": float}, ...]
    """
    def moderate(chat):
        prompt = tokenizer.apply_chat_template(chat, return_tensors="pt", tokenize=False)
        # Skip the generation of whitespace.
        # Now the next predicted token will be either "safe" or "unsafe"
        prompt += "\n\n"
        
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            output_logits=True,  # get logits
        )
        input_length = inputs.input_ids.shape[1]
        generated_token_ids = outputs.sequences[:, input_length:].cpu()
        generated_text = tokenizer.decode(generated_token_ids[0], skip_special_tokens=True)
        first_token_logits = outputs.logits[0]
        first_token_probs = torch.softmax(first_token_logits, dim=-1)
        unsafe_probability = first_token_probs[0, unsafe_token_id]
        unsafe_probability = unsafe_probability.item()
        return {
            "unsafe_score": unsafe_probability,
            "generated_text": generated_text
        }
    results = []
    for start_idx in tqdm(range(len(conversations)), desc="Batch Inference"):
        results.append(moderate(conversations[start_idx]))

    return results

def main():
    parser = argparse.ArgumentParser(description="Run batched moderation checks using a Llama model.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-Guard-3-8B",
                        help="Hugging Face model ID or path to a local model directory.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output JSON file.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference.")
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map='cuda'
    )

    # Read data
    data = read_jsonl(args.input_file)
    # We'll extract the 'messages' portion from each sample
    conversations = [item['messages'] for item in data]

    # Get token ID for "unsafe"
    unsafe_token_id = tokenizer.convert_tokens_to_ids("unsafe")
    safe_token_id = tokenizer.convert_tokens_to_ids("safe")
    print(f"Unsafe token id: {unsafe_token_id}")
    print(f"safe token id: {safe_token_id}")
    # Run batched moderation
    moderations = moderate_batch(
        model=model,
        tokenizer=tokenizer,
        conversations=conversations,
        unsafe_token_id=unsafe_token_id,
        safe_token_id = safe_token_id,
        batch_size=args.batch_size,
        device='cuda'
    )

    # Save results as JSON
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(moderations, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
