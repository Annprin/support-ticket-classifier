FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

# Install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code + configs
COPY src ./src
COPY config.yaml ./config.yaml

# Copy model artifacts (must exist locally before build)
COPY models ./models

ENTRYPOINT ["python", "-m", "src.predict"]
