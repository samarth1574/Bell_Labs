#!/bin/bash
echo "Starting SKU-110K UI and Backend..."

# Activate virtual environment
source .venv/bin/activate

# Start the custom backend in the background
echo "Starting backend on port 8000..."
PYTHONPATH=. python3 src/api.py &
BACKEND_PID=$!

# Wait for backend to bind
sleep 2

# Start the Vite frontend
echo "Starting frontend on port 5173..."
cd ui && npm run dev &
FRONTEND_PID=$!

echo "========================================="
echo "Backend running at:  http://localhost:8000"
echo "Frontend running at: http://localhost:5173"
echo "Press Ctrl+C to stop both servers."
echo "========================================="

# Trap Ctrl+C to kill both background processes
trap "kil