import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncpg
import ssl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Lexomat Intelligence API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ThreadPool for blocking model calls
executor = ThreadPoolExecutor(max_workers=2)

# Database pool
db_pool: asyncpg.pool.Pool | None = None

# Model (lazy-loaded)
model = None

# -------------------------------
# Database startup/shutdown
# -------------------------------
async def startup_db_pool():
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        logger.warning("SUPABASE_DB_URL not set. DB disabled.")
        return None

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE  # Only for testing

    max_retries = 5
    retry_delay = 2  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Database connection attempt {attempt}/{max_retries}")
            pool = await asyncpg.create_pool(
                dsn=db_url,
                min_size=1,
                max_size=5,
                ssl=ssl_context,
                command_timeout=60,
                timeout=10  # Connection timeout
            )
            # Test connection
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            logger.info("Database pool created successfully")
            return pool
        except Exception as e:
            logger.warning(f"Database connection attempt {attempt} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Failed to create database pool after {max_retries} attempts")
                return None

@app.on_event("startup")
async def startup():
    global db_pool
    logger.info("Starting up... Initializing database pool")
    db_pool = await startup_db_pool()
    if db_pool:
        logger.info("Application startup complete with database")
    else:
        logger.warning("Application started without database connection")

@app.on_event("shutdown")
async def shutdown():
    global db_pool
    if db_pool:
        logger.info("Shutting down... Closing database pool")
        await db_pool.close()

# -------------------------------
# Pydantic request model
# -------------------------------
class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"

# -------------------------------
# Model embedding
# -------------------------------
async def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model loaded successfully")
    return model

async def get_embedding(text: str):
    model_instance = await get_model()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, model_instance.encode, text)

# -------------------------------
# Health check endpoints
# -------------------------------
@app.get("/")
async def health_check():
    db_status = "connected" if db_pool else "disconnected"
    return {
        "status": "healthy",
        "message": "API running",
        "database": db_status,
        "endpoints": ["/search"]
    }

@app.get("/health")
async def health():
    # Always return healthy to pass Railway health check
    # Even if DB is not connected yet, the service is running
    return {"status": "healthy"}

# -------------------------------
# Search endpoint
# -------------------------------
@app.post("/search")
async def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available.")

    try:
        # Get embedding
        query_embedding = await get_embedding(req.query)
        query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        async with db_pool.acquire() as conn:
            if req.mode == "keyword":
                sql = """
                    SELECT id, title, body,
                    ts_rank(to_tsvector('english', title || ' ' || body), plainto_tsquery('english', $1)) AS fts_score
                    FROM documents
                    WHERE to_tsvector('english', title || ' ' || body) @@ plainto_tsquery('english', $1)
                    ORDER BY fts_score DESC
                    LIMIT 10;
                """
                results = await conn.fetch(sql, req.query)

            elif req.mode == "semantic":
                sql = """
                    SELECT id, title, body,
                    1 - (embedding <=> $1::vector) AS vector_score
                    FROM documents
                    ORDER BY vector_score DESC
                    LIMIT 10;
                """
                results = await conn.fetch(sql, query_embedding_str)

            else:  # hybrid
                sql = """
                    SELECT id, title, body,
                    ts_rank(to_tsvector('english', title || ' ' || body), plainto_tsquery('english', $1)) AS fts_score,
                    1 - (embedding <=> $2::vector) AS vector_score
                    FROM documents
                    ORDER BY 0.5 * COALESCE(ts_rank(to_tsvector('english', title || ' ' || body), plainto_tsquery('english', $1)), 0) 
                             + 0.5 * (1 - (embedding <=> $2::vector)) DESC
                    LIMIT 10;
                """
                results = await conn.fetch(sql, req.query, query_embedding_str)

        output = []
        for r in results:
            row = {"id": r["id"], "title": r["title"], "body": r["body"]}
            if "fts_score" in r:
                row["fts_score"] = float(r["fts_score"]) if r["fts_score"] else 0.0
            if "vector_score" in r:
                row["vector_score"] = float(r["vector_score"])
            output.append(row)

        return {"results": output, "count": len(output)}

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")