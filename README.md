# The Ultimate CheatSheet RAG System

## Description
This repository hosts an advanced Retrieval-Augmented Generation (RAG)
system. It combines a robust vector-based retrieval mechanism with a large language model,
enabling **accurate** and **context-aware** answers to user queries.

It supports a variety of file formats, notably PDFs (including potential
images or scanned content). The pipeline can automatically extract text,
generate image captions (if needed), and embed the information for quick
and relevant retrieval when answering questions about Excel features.

We rely on a [Kaggle dataset](https://www.kaggle.com/datasets/timoboz/data-science-cheat-sheets)
with various topics (DevOps, Data Science, Algorithms, etc.), extracting
structured information and building a comprehensive knowledge base.

## Key Components of This RAG Pipeline
1. **PDF & Image Processing**: 
   - Utilizes PyMuPDF to extract text from PDF pages.  
   - Converts pages or embedded images to bitmaps for embedding with a vision model if desired.
2. **Embeddings**: 
   - Uses ColPali (for instance) to generate vector representations for both text and images.
3. **Indexing in Milvus**: 
   - Stores the resulting embeddings in a Milvus collection for rapid vector search.
4. **Retrieval & Answer Generation**: 
   - On receiving a query (e.g. “How do I add a comment in Excel?”), the system searches Milvus for top-matching chunks.  
   - The best matches are passed to a large language model (e.g. Qwen2-VL), which uses the context to produce an answer.

## Used Tools

- **Vector Database**: Milvus  
- **Embeddings**: ColPali for text/image  
- **Text Generation**: Qwen2-VL (2B params) or similar LLM  
- **OCR/PDF**: PyMuPDF (fitz)  
- **Image Handling**: PIL  

## Example Workflow
![Example Implementation](https://github.com/Kerysfel/ITMO-ANLP/blob/main/answer.png)

1. **Load Excel PDF**  
2. **Extract Text** (and images if present)  
3. **Compute Embeddings** for text and/or images  
4. **Store** in Milvus  
5. **User Query**: “How can I add a comment without changing cell contents?”  
6. **Retrieve** relevant context from vector store  
7. **Generate** final answer (Qwen2-VL)

## Validation
A set of 20 questions (covering topics like DevOps, Data Science, SQL,
and more) was used for evaluation. We measured both the average
response time and the proportion of correctly answered queries.  

- **Average Time for Answer**: ~18,5 seconds  
- **Accuracy**: 68% of the questions received satisfactory answers.

## TO DO:
- [ ] Expand coverage for more Excel functionalities (pivot tables, macros, etc.)  
- [ ] Integrate Tesseract OCR if scanned PDFs are present  
- [ ] Improve chunking & summarization for large or complex cheat sheets  
- [ ] Add a more detailed evaluation framework (BLEU, ROUGE, etc.)  
- [ ] Provide a user-friendly web interface for PDF upload and querying  
