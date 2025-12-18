FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy shared euchre_core package
COPY shared/euchre_core /shared/euchre_core

# Copy requirements and install Python dependencies
COPY services/euchre-api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install shared euchre_core package
RUN pip install -e /shared/euchre_core

# Copy application code
COPY services/euchre-api/ .

# Expose port
EXPOSE 5000

# Run Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
