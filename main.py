import uvicorn
import os
from api import app # This imports the 'app' from your api.py

if __name__ == "__main__":
    # Railway sets a 'PORT' environment variable automatically
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)