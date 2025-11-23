#!/bin/bash

# --- Configuration Variables ---
IMAGE_NAME="rag-finance-chatbot"
CONTAINER_NAME="rag-finance-container"
HOST_PORT="8000"
CONTAINER_PORT="8000"

# --- REQUIRED: Replace with your actual API Key ---
# For security, you should load this from a .env file or environment,
# but for a quick script, setting it here works.
OPENAI_API_KEY=""
echo "--- 🏗️  Pulling repo"
git pull origin main
echo "--- 🏗️  Step 1: Building the Docker Image ($IMAGE_NAME) ---"
# Build the image. This is necessary if Dockerfile or requirements.txt changed.
# The `|| exit 1` stops the script if the build fails.
docker build -t "$IMAGE_NAME" . || exit 1

echo "--- 🛑 Step 2: Stopping and Removing Old Container (if exists) ---"
# Stop and remove the old container to avoid conflicts
if [ "$(docker ps -a -q -f name="$CONTAINER_NAME")" ]; then
    echo "Existing container found. Stopping and removing..."
    docker stop "$CONTAINER_NAME"
    docker rm "$CONTAINER_NAME"
fi

echo "--- 🚀 Step 3: Running the New Container with Bind Mount ---"
# The bind mount (-v) links the current directory ($PWD) to /app inside the container.
# This enables Uvicorn's --reload feature for instant code updates.
docker run -d \
  --name "$CONTAINER_NAME" \
  -p "$HOST_PORT":"$CONTAINER_PORT" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -v "${PWD}:/app" \
  "$IMAGE_NAME"

# Check if the container started successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================================="
    echo "✅ SUCCESS! Application is running in the background."
    echo "   - Container ID: $(docker ps -q -f name="$CONTAINER_NAME")"
    echo "   - Local URL: http://localhost:$HOST_PORT/docs"
    echo "=========================================================="
else
    echo "=========================================================="
    echo "❌ ERROR: Failed to start the Docker container."
    echo "   Check logs with: docker logs $CONTAINER_NAME"
    echo "=========================================================="
    exit 1
fi

echo ""
echo "--- 📄 Step 4: Displaying Initial Logs ---"
# Show the first few lines of logs to confirm Uvicorn started
docker logs "$CONTAINER_NAME" | tail -n 5