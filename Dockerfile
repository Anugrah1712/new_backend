# Use slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Non-interactive to avoid prompts
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies required by Chromium
RUN apt-get update && apt-get install -y \
    wget curl unzip \
    fonts-liberation libnss3 libnspr4 libdbus-1-3 \
    libxkbcommon0 libasound2 libjpeg-dev libxshmfence1 \
    libatk1.0-0 libatk-bridge2.0-0 libcups2 libpango-1.0-0 \
    libcairo2 libatspi2.0-0 xdg-utils libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libx11-xcb1 libx11-6 libxcb1 \
    --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and browsers (Chromium)
RUN playwright install --with-deps chromium

# Copy the full project
COPY . .

# Set environment variable for FastAPI port
ENV PORT 8000
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
