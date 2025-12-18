FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy shared euchre_core package and install it
COPY shared/euchre_core /tmp/euchre_core
RUN pip install -e /tmp/euchre_core

# Copy requirements and install Python dependencies
COPY services/ai-trainer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/ai-trainer/ .

# Copy and set entrypoint script
COPY services/ai-trainer/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose port
EXPOSE 5002

# Set PYTHONPATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Run Flask app
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5002"]
