#!/usr/bin/env python3
"""Startup script for the Open Deep Research FastAPI server.

This script provides a convenient way to start the FastAPI server with proper
configuration and logging.
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """Start the FastAPI server."""
    # Ensure the src directory is in the Python path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    print(src_path)
    print("Starting Open Deep Research FastAPI server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation will be available at: http://localhost:8000/docs")
    print("Alternative docs at: http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop the server")
    
    # Start the server
    uvicorn.run(
        "open_deep_research.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )



if __name__ == "__main__":
    main()