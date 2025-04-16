FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    unzip \
    git \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libxss1 \
    libasound2 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm-dev \
    libxshmfence-dev \
    fonts-liberation \
    libappindicator1 \
    libappindicator3-1 \
    xdg-utils \
    ca-certificates \
    xvfb && \
    apt-get clean

# Set work directory
WORKDIR /app

# Copy your requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install --with-deps

# Copy the rest of your code
COPY . .

# Expose the port if you're running an API
EXPOSE 8000

# Default command with xvfb-run
CMD ["xvfb-run", "--server-args=-screen 0 1024x768x24", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
