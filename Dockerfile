FROM tensorflow/tensorflow:2.15.0  

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your training script
COPY train_cnn.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the training
CMD ["python3", "train_cnn.py"]