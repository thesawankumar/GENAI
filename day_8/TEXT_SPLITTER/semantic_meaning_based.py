# from langchain_text_splitters import SemanticChunker
# from langchain_openai.embeddings import OpenAIEmbeddings
# from dotenv import load_dotenv

# load_dotenv()

# text_splitter = SemanticChunker(
#     OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation",
#     breakpoint_threshold_amount=3
# )

# sample = """
# Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


# Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
# """

# docs = text_splitter.create_documents([sample])
# print(len(docs))
# print(docs)





from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=40,
    separators=["\n\n", "\n", ".", " ", ""]
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds.
The sun was bright, and the air smelled of earth.

The Indian Premier League (IPL) is the biggest cricket league in the world.
People all over the world watch the matches.

Terrorism is a big danger to peace and safety.
It causes fear and destruction.
"""

docs = text_splitter.create_documents([sample])

print(len(docs))
for i, d in enumerate(docs):
    print(f"\n--- Chunk {i+1} ---")
    print(d.page_content)
