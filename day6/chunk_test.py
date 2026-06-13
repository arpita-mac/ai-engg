from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import  RecursiveCharacterTextSplitter

loader=PyPDFLoader("company_handbook.pdf")
pages=loader.load()
print(f"Number of pages loaded: {len(pages)}")
print(f"\n--- Raw content of page 1 (first 500 chars) ---")
print(pages[0].page_content[:500])

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # roughly how many characters per chunk
    chunk_overlap=50     # chunks overlap by this many characters
)

chunks = splitter.split_documents(pages)

print(f"\n\nNumber of chunks created: {len(chunks)}")
print(f"\n--- Chunk 1 ---")
print(chunks[0].page_content)
print(f"\n--- Chunk 2 ---")
print(chunks[1].page_content)
# extract just the text content from each chunk
chunk_texts = [chunk.page_content for chunk in chunks]

print(f"\n\nFirst chunk as plain text:")
print(chunk_texts[0])
print(f"\nTotal chunks ready for embedding: {len(chunk_texts)}")