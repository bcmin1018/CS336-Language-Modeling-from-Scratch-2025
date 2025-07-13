

import multiprocessing
import os
import time
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def count_pairs_worker(tokens_list):
    local_counts = {}
    for tokens in tokens_list:
        for pair in [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]:
            local_counts[pair] = local_counts.get(pair, 0) + 1
    return local_counts

def replace_best_pair_worker(args):
    tokens, best_pair, new_token = args
    j = 0
    new_tokens = []
    while j < len(tokens):
        if j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == best_pair:
            new_tokens.append(new_token)
            j += 2
        else:
            new_tokens.append(tokens[j])
            j += 1
    return new_tokens

def process_chunk(filename, start, end, split_token="<|endoftext|>"):
    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # Optional: split by special tokens and remove them
    docs = chunk.split(split_token)
    docs = [doc.strip() for doc in docs if doc.strip()]

    # Here, apply your own pre-tokenization logic (e.g., regex, cleaning)
    pretokenized = [doc for doc in docs]  # dummy

    return pretokenized


def run_serial_tokenization(filename: str):
    num_processes = os.cpu_count()
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    chunks = [(filename, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

    all_results = [process_chunk(*chunk) for chunk in chunks]

    # Flatten the list
    flattened = [doc for result in all_results for doc in result]
    return flattened

def run_parallel_tokenization(filename: str):
    num_processes = os.cpu_count() // 2
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    chunks = [(filename, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with multiprocessing.Pool(num_processes) as pool:
        all_results = pool.starmap(process_chunk, chunks)

    # Flatten the list
    flattened = [doc for result in all_results for doc in result]
    return flattened

if __name__ == "__main__":
    start = time.time()
    run_serial_tokenization('/Users/byeongcheolmin/PycharmProjects/llm_from_scratch/data/TinyStoriesV2-GPT4-train.txt')
    print(f"Serial tokenization took {1000*(time.time()-start):.2f} ms")

    start = time.time()
    run_parallel_tokenization('/Users/byeongcheolmin/PycharmProjects/llm_from_scratch/data/TinyStoriesV2-GPT4-train.txt')
    print(f"Parallel tokenization took {1000*(time.time()-start):.2f} ms")
    