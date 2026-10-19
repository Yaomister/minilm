import json
import regex as re
from config import Config
from collections import Counter
from utils.directories import PRETRAINING_RAW_DATASET_DIR, PRETRAINING_MERGES_DIR


class Tokenizer:
    """BPE Tokenizer."""

    def __init__(self):
        self.vocabulary = None
        self.merges = None

        # the regex for splitting up the text so that we dont end up with awkward tokens
        self.regex = re.compile(
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        )

        self.special_tokens = {"<|endoftext|>": Config.vocab_size - 1}

    def train(self, text, save_path):
        """Train the tokenizer."""

        print("training tokenizer")

        # split up the text
        chunks = self.regex.findall(text)
        chunks = [list(chunk.encode("utf-8")) for chunk in chunks]

        merges = {}

        # start from 256 and keep merging till the vocab size
        for new_id in range(256, Config.vocab_size - 1):
            counts = Counter()
            for chunk in chunks:
                for pair in zip(chunk, chunk[1:]):
                    counts[pair] += 1

            if not counts:
                break

            # get the most common tokens
            to_merge = max(counts, key=counts.get)

            if not counts or counts[to_merge] < 2:
                break

            # merge all existing pairs in the vocabulary
            chunks = [self._merge(chunk, to_merge, new_id) for chunk in chunks]

            merges[to_merge] = new_id

            if new_id % 100 == 0:
                print(f"training up to token {new_id}")

        json.dump({f"{a},{b}": v for (a, b), v in merges.items()}, open(save_path, "w"))

    def _merge(self, chunk, to_merge, new_id):
        new_chunk = []
        i = 0
        while i < len(chunk):
            # match the target pair
            if (
                i + 1 < len(chunk)
                and chunk[i] == to_merge[0]
                and chunk[i + 1] == to_merge[1]
            ):
                # merge it
                new_chunk.append(new_id)
                i += 2
            else:
                # keep original
                new_chunk.append(chunk[i])
                i += 1
        return new_chunk

    def load(self, load_path):
        """Load the trained tokenizer merges."""
        # already saved in order
        with open(load_path, "r", encoding="utf-8") as f:
            raw_merges = json.loads(f.read())
            merges = {tuple(map(int, k.split(","))): v for k, v in raw_merges.items()}

        vocabulary = {i: bytes([i]) for i in range(256)}

        for pair, id in merges.items():
            vocabulary[id] = vocabulary[pair[0]] + vocabulary[pair[1]]

        for token, id in self.special_tokens.items():
            vocabulary[id] = token.encode("utf-8")

        self.merges = merges
        self.vocabulary = vocabulary

    def encode(self, text):
        """Turn text into tokens."""
        assert self.merges is not None
        pattern = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
        ids = []
        for part in re.split(pattern, text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                ids.extend(self._encode(part))

        return ids

    def _encode(self, text):
        ids = []
        for chunk in self.regex.findall(text):
            chunk_ids = list(chunk.encode("utf-8"))
            while len(chunk_ids) >= 2:
                pair = min(
                    zip(chunk_ids, chunk_ids[1:]),
                    key=lambda pair: self.merges.get(pair, float("inf")),
                )
                if not pair in self.merges:
                    # when we have no more merged pairs in the vocab, we are done merging the tokens
                    break
                chunk_ids = self._merge(chunk_ids, pair, self.merges[pair])

            ids.extend(chunk_ids)
        return ids

    def decode(self, ids):
        """Turn tokens into text."""
        assert self.vocabulary is not None

        text = []

        for id in ids:
            # vocab_size is bigger than the learned vocabulary, a model can sample ids with no bytes
            text.append(self.vocabulary.get(id, b""))

        return b"".join(text).decode("utf-8", errors="replace")


if __name__ == "__main__":

    bpe = Tokenizer()

    with open(PRETRAINING_RAW_DATASET_DIR, "r", encoding="utf-8") as f:
        text = f.read()

    bpe.train(text, PRETRAINING_MERGES_DIR)
