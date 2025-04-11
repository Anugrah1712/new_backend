# Use a slim Python image to reduce memory footprint
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set non-interactive mode
ARG DEBIAN_FRONTEND=noninteractive

# Install only essential dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    fonts-liberation \
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libxkbcommon0 \
    libasound2 \
    libjpeg-dev \
    libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

# Install virtualenv and create a venv
RUN python -m venv /app/venv

# Activate virtual environment
ENV PATH="/app/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Run crawl4ai setup script (installs Playwright browser deps)
RUN /app/venv/bin/python -m crawl4ai.setup

# Copy the entire project
COPY . .

# Use dynamic port for Render
ENV PORT 8000
EXPOSE 8000

# Start FastAPI app with a single worker to reduce memory usage
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
