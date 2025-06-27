FROM python:3.10-slim

# Install system dependencies (including SSL, DNS, and certificate tools)
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
    libxml2-dev \
    libxslt-dev \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    zlib1g-dev \
    python3-dev \
    apt-transport-https \
    gnupg \
    tzdata \
    && apt-get clean

# Set the working directory
WORKDIR /app

# Copy Python requirements
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and its browsers
RUN pip install playwright && playwright install --with-deps

# Install Scrapy and Scrapy-Splash
RUN pip install scrapy scrapy-splash

# Install FastText (separately due to compilation)
RUN pip install fasttext

# Download FastText language model
RUN wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz

# Copy project files into the container
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
