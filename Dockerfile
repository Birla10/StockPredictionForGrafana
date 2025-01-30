FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the script and requirements into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your script uses (optional)
EXPOSE 8000

# Run the Python script
CMD ["python", "predict_stocks.py"]
