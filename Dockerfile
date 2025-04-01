# Use official Python image as base
FROM python:3.11

# Set working directory in the container
WORKDIR /recommenderApp

# Copy requirements.txt before installing dependencies (for caching efficiency)
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "st_bible.py", "--server.port=8501", "--server.address=0.0.0.0"]
