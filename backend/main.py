from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Handwriting AI backend is up and running!"}

@app.get("/info")
def info():
    return {"project": "Handwriting AI", "status": "Started"}
