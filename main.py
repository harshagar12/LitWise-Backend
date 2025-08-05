"""
FastAPI server for LitWise recommendation engine
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from recommendation_engine import recommendation_engine
except ImportError as e:
    print(f"Warning: Could not import recommendation engine: {e}")
    recommendation_engine = None

app = FastAPI(title="LitWise Recommendation API", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class GenreRequest(BaseModel):
    top_n: int = 20

class ClusterRequest(BaseModel):
    selected_tag_ids: List[int]
    num_clusters: int = 3

class RecommendationRequest(BaseModel):
    favorite_goodreads_book_ids: List[int]
    top_n: int = 10

@app.get("/")
async def root():
    return {"message": "LitWise Recommendation API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "engine_loaded": recommendation_engine is not None,
        "data_loaded": recommendation_engine.data_loaded if recommendation_engine else False
    }

@app.post("/api/python/genres")
async def get_genres(request: GenreRequest):
    if not recommendation_engine:
        raise HTTPException(status_code=500, detail="Recommendation engine not available")
    
    try:
        genres = recommendation_engine.get_genres(request.top_n)
        return genres
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting genres: {str(e)}")

@app.post("/api/python/clusters")
async def get_clusters(request: ClusterRequest):
    if not recommendation_engine:
        raise HTTPException(status_code=500, detail="Recommendation engine not available")
    
    try:
        clusters = recommendation_engine.get_book_clusters(
            request.selected_tag_ids, 
            request.num_clusters
        )
        return clusters
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting clusters: {str(e)}")

@app.post("/api/python/recommendations")
async def get_recommendations(request: RecommendationRequest):
    if not recommendation_engine:
        raise HTTPException(status_code=500, detail="Recommendation engine not available")
    
    try:
        recommendations = recommendation_engine.get_recommendations(
            request.favorite_goodreads_book_ids,
            request.top_n
        )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting LitWise Recommendation API...")
    print("Visit http://localhost:8000/docs for API documentation")
    uvicorn.run(app, host="0.0.0.0", port=8000)
