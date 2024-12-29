# Use the official Python image
FROM python:3.9-slim

# Create a working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY app2.py .

# We expose port 8080 for documentation (though Cloud Run configures runtime PORT envvar)
EXPOSE 8080

# Set the entrypoint to run the Flask app
CMD ["python", "app2.py"]
