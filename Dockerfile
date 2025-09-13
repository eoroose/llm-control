# python/Dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies needed for compiling
RUN apt-get update --fix-missing && \
    apt-get install -y --fix-missing build-essential git && \
    apt-get clean

# Copy only the Python project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
