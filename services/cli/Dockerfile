FROM python:3.11-slim

WORKDIR /app

# Copy shared euchre_core package
COPY shared/euchre_core /shared/euchre_core

# Copy requirements and install Python dependencies
COPY services/cli/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install shared euchre_core package
RUN pip install -e /shared/euchre_core

# Copy application code
COPY services/cli/ .

# Run CLI application
CMD ["python", "app/main.py"]
