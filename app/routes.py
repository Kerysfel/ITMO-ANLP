from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.embeddings import process_and_store_pdf
from app.agent import RAGAgent

router = APIRouter()
agent = RAGAgent()

@router.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload and process a PDF.
    It stores embeddings (text + images) in Milvus.
    """
    try:
        await process_and_store_pdf(file)
        return {"message": "PDF successfully uploaded and processed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")

@router.post("/query/")
async def query_cheat_sheet(query: str = Form(...)):
    """
    Endpoint to query the cheat sheets using RAGAgent.
    """
    try:
        response = await agent.run(query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")