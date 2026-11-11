# Official Ultralytics image for NVIDIA Jetson (JetPack 6).
# PyTorch, OpenCV, NumPy and Ultralytics come with the image and are built to work together.
FROM ultralytics/ultralytics:latest-jetson-jetpack6

RUN pip install --no-cache-dir streamlit==1.65.0 pytest

WORKDIR /app
COPY . .
EXPOSE 8501
CMD ["python3", "-m", "streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
