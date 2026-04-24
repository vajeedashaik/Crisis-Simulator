FROM python:3.11-slim

WORKDIR /app

COPY schema.py environment.py rewards.py agents.py curriculum.py app.py ./

RUN pip install --no-cache-dir openenv fastapi uvicorn pydantic

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
