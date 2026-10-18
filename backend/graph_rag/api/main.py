"""uvicorn entry point: ``uvicorn graph_rag.api.main:app``."""

from graph_rag.api.app import create_app

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("graph_rag.api.main:app", host="0.0.0.0", port=8000, reload=True)
