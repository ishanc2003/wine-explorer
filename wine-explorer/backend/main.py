import asyncio
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import analyst
import llm
import rag
import transcribe

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

@asynccontextmanager
async def lifespan(app: FastAPI):
    transcribe.load_model()
    rag.build_index()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    try:
        text = await asyncio.to_thread(transcribe.transcribe, audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"text": text}

class QueryRequest(BaseModel):
    text: str

@app.post("/query")
async def query(req: QueryRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Query text is empty.")

    analytic_result: str | None = None
    context: list[str] = []
    if analyst.is_analytical(req.text):
        analytic_result = await asyncio.to_thread(analyst.run, req.text, rag._df)

    if analytic_result is None:
        try:
            context = await asyncio.to_thread(rag.retrieve, req.text)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[RETRIEVAL ERROR]\n{tb}")
            raise HTTPException(status_code=500, detail=str(e))

    async def generate():
        try:
            async for chunk in llm.ask_stream(req.text, context, analytic_result):
                yield chunk
        except Exception:
            print(f"[STREAMING ERROR]\n{traceback.format_exc()}")
            raise

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
