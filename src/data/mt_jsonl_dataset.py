import json
from torch.utils.data import Dataset


class MTForgetJSONLDataset(Dataset):
    """Loads multi-turn forget conversations from a JSONL file.

    Each line must be a JSON object with a 'conversation' key (list of
    {role, content} dicts) and optionally 'question', 'answer',
    'attack_type' keys.  Items are returned as raw dicts; tokenization
    happens downstream in MTForgetRetainDataset.__init__.
    """

    def __init__(self, jsonl_path: str, **kwargs):
        super().__init__()
        self.data = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
