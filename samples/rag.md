# Retrieval Augmented Generation

RAG combines retrieval with generation. A retrieval system first finds relevant pieces of a knowledge base, then those pieces are provided to a language model as context for generating an answer.

Documents are often split into smaller chunks before embedding. Smaller chunks can make retrieval more precise because a single vector does not have to represent an entire long document.

Chunk overlap keeps some text from the previous chunk at the beginning of the next one. This can preserve context when an important sentence crosses a chunk boundary.

Embedding search usually returns chunks rather than whole documents. Metadata such as document ID and source filename can be stored alongside each chunk so the application can identify where retrieved text came from.
