import json
import torch
from torch.utils.data import Dataset

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            data.append(json_object)
    return data

def concat_messages(messages, tokenizer):
    """
    Concatenates a list of messages into a single text string using special role-based markers.

    Args:
        messages (List[dict]): List of chat-format messages.
        tokenizer: Hugging Face tokenizer.

    Returns:
        str: Formatted chat-format string.
    
    Raises:
        ValueError: If an invalid role is found in messages.
    """
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text

def encode_with_messages_format(example, tokenizer, max_seq_length):
    """
    Tokenizes a chat-format example and masks non-assistant tokens in the label.

    Args:
        example (dict): A dictionary with a 'messages' key containing the chat history.
        tokenizer: A Hugging Face tokenizer.
        max_seq_length (int): Maximum sequence length for tokenization.

    Returns:
        dict: A dictionary containing tokenized input_ids, labels, and attention_mask.
    
    Raises:
        ValueError: If the example contains no messages.
    """
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError("Example has no 'messages'.")

    example_text = concat_messages(messages, tokenizer)
    tokenized_example = tokenizer(example_text, 
                                  return_tensors='pt', 
                                  max_length=max_seq_length, 
                                  truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                so_far_text = concat_messages(messages[:message_idx], tokenizer)
                message_start_idx = tokenizer(
                    so_far_text,
                    return_tensors='pt',
                    max_length=max_seq_length,
                    truncation=True
                ).input_ids.shape[1]

            if (message_idx < len(messages) - 1 
                and messages[message_idx+1]["role"] == "assistant"):
                messages_so_far = concat_messages(messages[:message_idx+1], tokenizer) + "<|assistant|>\n"
            else:
                messages_so_far = concat_messages(messages[:message_idx+1], tokenizer)
            
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]

            labels[:, message_start_idx:message_end_idx] = -100
            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }

class MessageDataset(Dataset):
    """
    PyTorch Dataset for chat-style formatted messages for dattri.
    """
    def __init__(self, list_of_examples, tokenizer, max_length=512):
        super().__init__()
        self.data = list_of_examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        encoded = encode_with_messages_format(example, self.tokenizer, self.max_length)
        return (
            encoded["input_ids"],
            encoded["labels"]
        )

class MessageDatasetRepSim(Dataset):
    def __init__(self, list_of_examples, tokenizer, max_length=512):
        super().__init__()
        self.data = list_of_examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        encoded = encode_with_messages_format(example, self.tokenizer, self.max_length)
        return {
            "input_ids": encoded["input_ids"],
            "labels": encoded["labels"],
            "attention_mask": encoded["attention_mask"]
        }
