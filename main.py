"""
Main entrypoint to run the IRIS backend.
"""

import uvicorn
from config import settings

if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower()
    )
