FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy shared euchre_core package and install it
COPY shared/euchre_core /tmp/euchre_core
RUN pip install -e /tmp/euchre_core

# Copy requirements and install Python dependencies
COPY services/web-ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/web-ui/ .

# Copy and set entrypoint script
COPY services/web-ui/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose port
EXPOSE 5001

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Run Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]
