from fastapi import FastAPI

app = FastAPI(title="TailorResume AI Service", version="0.1.0")


@app.get("/health")
def health_check() -> dict:
    return {"ok": True}
