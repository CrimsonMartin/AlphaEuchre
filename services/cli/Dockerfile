FROM python:3.11-slim

WORKDIR /app

# Copy shared euchre_core package and install it
COPY shared/euchre_core /tmp/euchre_core
RUN pip install -e /tmp/euchre_core

# Copy requirements and install Python dependencies
COPY services/cli/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/cli/ .

# Copy and set entrypoint script
COPY services/cli/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Run CLI application
CMD ["python", "app/main.py"]
