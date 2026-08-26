# Use official ultra-fast uv image with Python 3.12 slim
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Enable bytecode compilation for performance
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy dependency definition
COPY pyproject.toml poetry.lock ./

# Install project dependencies
RUN uv pip install --system -r pyproject.toml

# Copy project source code and templates
COPY src ./src
COPY templates ./templates
COPY README.md ./

# Create directories for persistent storage
RUN mkdir -p data uploads chroma_db

# Set Python path so `src` modules can be imported directly
ENV PYTHONPATH=/app/src

# Expose FastAPI server port
EXPOSE 8080

# Start FastAPI application
CMD ["python", "src/app/app.py"]
