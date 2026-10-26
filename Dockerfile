# NVIDIA's ML image for JetPack 6 (same image as the course container "jetson_labs_dev").
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
