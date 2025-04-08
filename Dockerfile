FROM nvcr.io/nvidia/pytorch:22.04-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    klayout \
    verilator \
    gtkwave \
    libgl1

# Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Project setup
WORKDIR /app
COPY . .

# Entrypoint for training
CMD ["python", "scripts/train_model.py"]
