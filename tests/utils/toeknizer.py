from typing import Dict, List, Tuple, Iterable, Iterator, Optional, ClassVar
import regex as re
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        self.pet = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}
        self.gpt2_bytes_to_unicode = gpt2_bytes_to_unicode()
        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        self.id_to_byte = {k: v for k, v in self.vocab.items()}
        self.special_tokens = special_tokens or []
        # Add special tokens to vocab if not present
        for token in self.special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes not in self.byte_to_id:
                new_id = max(self.vocab.keys()) + 1
                self.vocab[new_id] = token_bytes
                self.byte_to_id[token_bytes] = new_id
                self.id_to_byte[new_id] = token_bytes

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None):
        import json
        # Load vocab
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
        vocab = {int(k): bytes.fromhex(v) if isinstance(v, str) else bytes(v) for k, v in vocab_json.items()}
        # Load merges
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append((parts[0].encode('utf-8'), parts[1].encode('utf-8')))
        return cls(vocab, merges, special_tokens)




    def encode(self, text: str) -> List[int]:
        import regex as re
        # Special tokens 정렬 (길이가 긴 것부터)
        # 동일한 스페셜 토큰이 여러 개 있을 경우, 길이가 긴 것부터 정렬하여 처리
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)

        # 스페셜 토큰을 정규식에 걸리지 않는 특수 문자로 치환
        special_token_map = {}
        for i, token in enumerate(sorted_special_tokens):
            placeholder = f" {chr(0xE000 + i)}"  # 앞뒤에 공백 추가
            special_token_map[placeholder] = token  # 공백 제거된 버전으로 매핑
            text = text.replace(token, placeholder)

        # 일반 토큰화 진행 finditer을 통해 메모리 효율적으로 처리
        # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        pretokenized = []
        for match in re.finditer(self.pat, text):
            token = match.group()
            if token in special_token_map:
                token = special_token_map[token]
            pretokenized.append(token)

        # 치환된 스페셜 토큰을 다시 원래대로 복원
        for i, pre_token in enumerate(pretokenized):
            if pre_token in special_token_map:
                pretokenized[i] = special_token_map[pre_token]

        merges = {pair: i for i, pair in enumerate(self.merges)}
        final_tokens = []
        for idx, token in enumerate(pretokenized):
            if token in self.special_tokens:
                special_token_bytes = token.encode('utf-8')
                final_tokens.append(special_token_bytes)  # 스페셜 토큰은 그대로 바이트로 변환
                continue
            token_bytes = [bytes([b]) for b in token.encode('utf-8')]
            while True:
                pairs = [(token_bytes[i], token_bytes[i + 1]) for i in range(len(token_bytes) - 1)]
                # for i, pair in enumerate(pairs):
                #     if pair in merges:
                #         token_bytes[i:i + 2] = [token_bytes[i] + token_bytes[i + 1]]
                #         print(f"Pair merged: {pair} -> {token_bytes[i]}")
                merge_candidates = [(i, merges.get(pair)) for i, pair in enumerate(pairs) if pair in merges]
                if not merge_candidates:
                    break

                i, pair = min(merge_candidates, key=lambda x: x[1])
                token_bytes[i:i+2] = [token_bytes[i] + token_bytes[i+1]]
        
            final_tokens.extend(token_bytes)

        ids = []
        for token in final_tokens:
            if token in self.special_tokens:
                token_id = self.byte_to_id[token]
            else:
                token_id = self.byte_to_id[token]
            ids.append(token_id)
        return ids
                
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for id_ in self.encode(text):
                yield id_

    def decode(self, ids: List[int]) -> str:
        # Map ids to bytes
        byte_seq = b''.join([self.id_to_byte.get(i, b'') for i in ids])
        # Decode bytes to string, replacing malformed bytes
        return byte_seq.decode('utf-8', errors='replace')


if __name__ == "__main__":
    import tiktoken
    from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
    VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
    MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    encoded_ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]

    reference_tokenizer = tiktoken.get_encoding("gpt2")
    reference_encoded_ids = reference_tokenizer.encode(test_string, allowed_special={"<|endoftext|>"})
    print(encoded_ids)
    print(reference_encoded_ids)