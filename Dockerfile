FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

WORKDIR /app

COPY agents/ graph/ llm/ mutators/ prompts/ selection/ utils/ .env app.py config.py main.py poetry.lock pyproject.toml /app/

RUN poetry config virtualenvs.create false
RUN poetry install

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]