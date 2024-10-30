# Stage 1: Base image with dependencies
FROM python:3.11-slim as base
LABEL maintainer="your-email@example.com"

# Set workdir
WORKDIR /app

# Avoid running as root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Set up the environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy requirements file to the image
COPY requirements.txt .

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y build-essential && \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

# Run the application as the created user from above
USER appuser

# Stage 2: Application Source Code
FROM base AS app

# Set workdir again
WORKDIR /app

# Copy source code
COPY . .

# Expose any necessary ports
EXPOSE 8075

# Set the entrypoint to start the app
ENTRYPOINT ["python"]
CMD ["app.py"]
