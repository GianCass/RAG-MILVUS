from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas import AskRequest, AskResponse
from app.services import rag as RAG
from app.settings import get_settings
import requests, json

router = APIRouter(tags=["rag"])

@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    s = get_settings()
    hits = RAG.search_similar(payload.query, top_k=payload.top_k or s.top_k)
    if not hits or min(h["score"] for h in hits) > (payload.abstain_threshold or s.abstain_threshold):
        return AskResponse(answer="no data", hits=hits)
    prompt = RAG.build_prompt(payload.query, hits)
    out = RAG.generate_answer(prompt)
    return AskResponse(answer=out.strip(), hits=hits)

@router.post("/ask/stream")
def ask_stream(payload: AskRequest):
    s = get_settings()
    hits = RAG.search_similar(payload.query, top_k=payload.top_k or s.top_k)
    if not hits or min(h["score"] for h in hits) > (payload.abstain_threshold or s.abstain_threshold):
        def gen():
            yield "data: no data\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    prompt = RAG.build_prompt(payload.query, hits)

    def stream():
        r = requests.post(f"{s.ollama_host}/api/generate",
                          json={"model": s.gen_model, "prompt": prompt, "stream": True},
                          stream=True)
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            if "response" in chunk:
                yield f"data: {chunk['response']}\n\n"
            if chunk.get("done"):
                break
    return StreamingResponse(stream(), media_type="text/event-stream")
