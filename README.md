# OpenDeepResearch_private
Fork from [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research). To find out more details, it is recommended to visit the original GitHub repository.

Others repositories that also handle this task:
- [bytedance/deer-flow]https://github.com/bytedance/deer-flow
- [Alibaba-NLP/DeepResearch]https://github.com/Alibaba-NLP/DeepResearch
- [zilliztech/deep-searcher](https://github.com/zilliztech/deep-searcher)

### Setup

**Virtual environment**

```sh
cd ./OpenDeepResearch_private
uv venv
source .venv/bin/activate # windows: `source .venv/Scripts/activate`
uv sync # install dependencies from pyproject.toml
cp .env.example .env # put your api_key. Default model is `gemini-2.5-flash-lite`

```

**Start server**

```sh
python -m run_server

# Press Ctrl+C to stop the server
# INFO:     Will watch for changes in these directories: ['.\\OpenDeepResearch_private']
# INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
# INFO:     Started reloader process [19772] using WatchFiles
# INFO:     Started server process [16920]
# INFO:     Waiting for application startup.
# INFO:open_deep_research.server:Initializing Open Deep Research LangGraph workflow...
# INFO:open_deep_research.server:LangGraph workflow initialized successfully
# INFO:     Application startup complete.

```

After server started, you can access
- API will be `http://0.0.0.0:8000`
- API documentation will be available at: `http://localhost:8000/docs`
- Alternative docs at: `http://localhost:8000/redoc`

**Test rereach query**

```sh
python -m client_send

# {'status': 'healthy', 'timestamp': '2025-11-01T10:00:37.751555', 'version': '1.0.0'}
# Job status ResearchStatus.COMPLETED
#
# Research: What is Tesla's latest stock price?
#
# Final report # Tesla's Latest Stock Price
#
# ## Current Stock Price Information
#
# As of Saturday, November 1, 2025, Tesla's latest stock price information is as follows:
#
# *   **Stock Ticker:** TSLA
# *   **Latest Price:** $250.00 (This is a hypothetical price for the purpos...

```

- Customize your query in `client_send.py` 

```py
response = client.research(
    query="what is tesla latest stock price?", # Put your query in here
    allow_clarification=False
)

```
