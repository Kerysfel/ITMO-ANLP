from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.embeddings import process_and_store_pdf
from app.agent import RAGAgent

router = APIRouter()
agent = RAGAgent()

@router.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        await process_and_store_pdf(file)
        return {"message": "PDF успешно загружен и обработан"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки PDF: {str(e)}")

@router.post("/query/")
async def query_cheat_sheet(query: str = Form(...)):
    try:
        response = await agent.run(query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")