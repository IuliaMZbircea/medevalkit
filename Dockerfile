FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY data/ data/

# Install the package
RUN pip install --no-cache-dir -e .

# Create directories for output
RUN mkdir -p reports /root/.medevalkit

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default to host.docker.internal for Ollama access
ENV OLLAMA_HOST=http://host.docker.internal:11434

ENTRYPOINT ["medeval"]
CMD ["run", "--n", "10"]