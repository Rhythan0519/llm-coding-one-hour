"""
Day 29: Text Chunking

Goal:
Split a long text into word chunks with overlap.

Input:
text: str
chunk_size: int
overlap: int

Output:
chunks: list[str]

Check:
Chunk count, max chunk length, overlap, empty input, invalid overlap.
"""
def chunk_text(text, chunk_size, overlap):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    words = text.split()

    if len(words) == 0:
        return []
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
    return chunks

def test():
    words = [f"w{i}" for i in range(20)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=6, overlap=2)

    assert isinstance(chunks, list)
    assert len(chunks) == 5
    assert all(len(chunk.split()) <= 6 for chunk in chunks)
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]
    assert chunk_text("", chunk_size=6, overlap=2) == []

    raised = False
    try:
        chunk_text(text, chunk_size=4, overlap=4)
    except ValueError:
        raised = True
    assert raised is True

    print("Input words:", len(words))
    print("Chunk count:", len(chunks))
    print("First chunk:", chunks[0])
    print("Test passed.")


if __name__ == "__main__":
    test()