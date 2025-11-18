FROM python:3.13-slim

# Set working directory
WORKDIR /next_bot

# Install system dependencies (minimal set — tu n'as pas besoin de wget sauf si tu l'utilises dans ton code)
RUN apt-get update && apt-get install -y gcc g++ && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt separately (pour tirer parti du cache Docker si le code change souvent)
COPY requirements.txt .

# Install Python dependencies
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy full app source
COPY . .

# Define default env vars
ENV PYTHONUNBUFFERED=1
ENV DEBUG_LOCAL=false

# Entrypoint (tu peux personnaliser ici si tu veux appeler via module : `python -m trade_bot.main`)
CMD ["python", "-m", "next_bot.app.run_replay"]

# 01 the command to create an image of the app for docker 

# docker build -t trading-bot:latest .
# docker tag trading-bot:latest 174175447862.dkr.ecr.ap-northeast-1.amazonaws.com/tradebot:latest
# aws ecr get-login-password --region ap-northeast-1 | docker login --username AWS --password-stdin 174175447862.dkr.ecr.ap-northeast-1.amazonaws.com
# docker push 174175447862.dkr.ecr.ap-northeast-1.amazonaws.com/tradebot:latest   
