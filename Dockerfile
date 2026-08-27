# Use a lightweight official Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install them securely
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API script and the trained models directory
COPY app.py .
COPY models/ ./models/

# Expose port 8000 for the Uvicorn server
EXPOSE 8000

# Command to run the API when the container starts
# We use 0.0.0.0 so it is accessible from outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]