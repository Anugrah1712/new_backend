#dockerfile

# Use an official Python image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Set non-interactive mode to avoid prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies required by Playwright
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    xvfb \
    fonts-liberation \
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libxkbcommon0 \
    libasound2 \
    libatspi2.0-0 \
    libappindicator3-1 \
    libjpeg-dev \
    libxshmfence1 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libwoff1 \
    libopus0 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m playwright install --with-deps

# Copy the entire project to the container
COPY . .

# Expose the port your app will run on
EXPOSE 8000

# Start the FastAPI app using xvfb-run
CMD ["xvfb-run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]

