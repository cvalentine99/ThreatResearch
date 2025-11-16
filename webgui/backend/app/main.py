"""FastAPI main application"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.core.config import settings
from app.api import upload, packets, flows, stats, stream, graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="GPU-accelerated PCAP analysis web interface",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": settings.APP_VERSION}


# Root endpoint
@app.get("/")
async def root():
    """API root"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }


# Include API routers
app.include_router(upload.router, prefix=f"{settings.API_V1_PREFIX}/upload", tags=["Upload"])
app.include_router(packets.router, prefix=f"{settings.API_V1_PREFIX}", tags=["Packets"])
app.include_router(flows.router, prefix=f"{settings.API_V1_PREFIX}", tags=["Flows"])
app.include_router(stats.router, prefix=f"{settings.API_V1_PREFIX}", tags=["Statistics"])
app.include_router(stream.router, prefix=f"{settings.API_V1_PREFIX}", tags=["Streaming"])
app.include_router(graph.router, prefix=f"{settings.API_V1_PREFIX}", tags=["Graph"])


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"OpenSearch URL: {settings.OPENSEARCH_URL}")
    logger.info(f"CUDA Parser: {settings.CUDA_PARSER_PATH}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("Shutting down application")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
