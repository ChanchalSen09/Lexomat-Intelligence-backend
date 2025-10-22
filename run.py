import os
import uvicorn
from main import app  # your FastAPI app

if __name__ == "__main__":
    # Get PORT from environment, default to 8000
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=4)
