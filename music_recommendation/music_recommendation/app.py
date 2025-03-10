from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pickle
import faiss
import numpy as np
import pandas as pd
import uvicorn
import os
import traceback

# Global variables to store our loaded models and data
svd_model = None
index = None
df = None
feature_matrix = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    global svd_model, index, df, feature_matrix
    
    try:
        # Load the SVD model
        print("Loading SVD model...")
        with open("svd_model.pkl", "rb") as f:
            svd_model = pickle.load(f)
        print("✅ SVD model loaded.")
        
        # Load the FAISS index
        print("Loading FAISS index...")
        index = faiss.read_index("faiss_index.bin")
        print("✅ FAISS index loaded.")
        
        # Load the track ID mapping
        print("Loading track ID mapping...")
        df = pd.read_csv("track_id_mapping.csv", index_col=0)
        print("✅ Track ID mapping loaded.")
        
        # Load the feature matrix for FAISS search
        print("Loading feature matrix...")
        feature_matrix = np.load("feature_matrix.npy")
        print("✅ Feature matrix loaded.")
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        # We'll continue even if models fail to load, but endpoints will fail
        
    yield  # This is where the app runs
    
    # Cleanup (when app is shutting down)
    print("Shutting down and cleaning up resources...")
    # Clear resources to free memory
    svd_model = None
    index = None
    df = None
    feature_matrix = None

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Song Recommendation API",
    description="API for recommending similar songs using FAISS",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class RecommendationResponse(BaseModel):
    track_name: str
    track_id: str
    recommendations: List[Dict[str, Any]]

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    error_detail = f"An unexpected error occurred: {str(exc)}"
    print(f"❌ Error: {error_detail}")
    print(f"Stack trace: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "details": error_detail},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with custom format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "details": None},
    )

def get_track_id_by_name(track_name):
    """
    Find a track ID based on the track name.
    Takes the first match if multiple tracks with the same name exist.
    
    Args:
        track_name (str): The name of the track to search for
        
    Returns:
        str: The track ID if found
        
    Raises:
        HTTPException: If track name is not found
    """
    if df is None:
        raise HTTPException(status_code=503, detail="Database not loaded. Please try again later.")
    
    # Case-insensitive search
    matches = df[df['track_name'].str.lower() == track_name.lower()]
    
    # If no exact match, try partial match
    if matches.empty:
        matches = df[df['track_name'].str.lower().str.contains(track_name.lower())]
    
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"No tracks found with name '{track_name}'")
    
    # Take the first match instead of raising an error for multiple matches
    # This allows the API to work with duplicate track names
    return matches['track_id'].iloc[0], matches['track_name'].iloc[0]

def hybrid_recommend(track_id, k=5):
    """
    Generates recommendations using FAISS (content-based filtering).
    Ensures recommendations are unique by track name.

    Args:
        track_id (str): The track ID for which recommendations are needed.
        k (int): Number of recommendations to return.

    Returns:
        list: A list of recommended song names and artists in JSON format.
    """
    # Check if models are loaded
    if df is None or index is None or feature_matrix is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please try again later.")
    
    # Check if track_id exists
    if track_id not in df['track_id'].values:
        raise HTTPException(status_code=404, detail=f"⚠️ Track ID '{track_id}' not found!")

    try:
        # Get track index for FAISS
        track_idx = df.index[df['track_id'] == track_id].tolist()[0]
        original_track_name = df.loc[track_idx, 'track_name']

        # Get similar tracks from FAISS - we'll need to fetch more than k to account for duplicates
        # Start with k*3 to have enough candidates after filtering
        fetch_count = k * 3
        query_vector = feature_matrix[track_idx].reshape(1, -1)
        
        # Get recommendations until we have enough unique ones
        unique_recommendations = []
        seen_track_names = set([original_track_name])  # Start with original track to exclude it
        
        while len(unique_recommendations) < k:
            distances, indices = index.search(query_vector, fetch_count)
            similar_songs = df.iloc[indices[0][1:]]  # Exclude original song
            
            # Filter for unique track names
            for _, song in similar_songs.iterrows():
                if song['track_name'] not in seen_track_names:
                    unique_recommendations.append({
                        'track_name': song['track_name'], 
                        'artists': song['artists']
                    })
                    seen_track_names.add(song['track_name'])
                    
                    if len(unique_recommendations) >= k:
                        break
            
            # If we've gone through all similar songs and still don't have enough unique ones,
            # we might need to increase our search radius
            if len(unique_recommendations) < k:
                fetch_count += k * 2
                if fetch_count > len(df):  # Avoid searching beyond dataset size
                    break
            else:
                break
                
        return unique_recommendations[:k]  # Return only requested number
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    models_loaded = all(x is not None for x in [svd_model, index, df, feature_matrix])
    return {
        "status": "healthy" if models_loaded else "degraded",
        "message": "API is running with all models loaded" if models_loaded else "API is running but models failed to load"
    }

@app.get("/recommend", response_model=RecommendationResponse, 
         responses={
             404: {"model": ErrorResponse}, 
             503: {"model": ErrorResponse},
             500: {"model": ErrorResponse}
         }, 
         tags=["Recommendations"])
async def recommend(
    track_name: Optional[str] = Query(None, description="The name of the track to get recommendations for"),
    track_id: Optional[str] = Query(None, description="The track ID to get recommendations for"),
    k: int = Query(5, description="Number of recommendations to return"),
    model: Optional[str] = Query("hybrid", description="Model type: hybrid, content, or collaborative")
):
    """
    Get song recommendations based on track name or track ID.
    
    - Provide either **track_name** OR **track_id** (if both are provided, track_id takes precedence)
    - **k**: Number of recommendations to return (default: 5)
    - **model**: Model type to use for recommendations (default: hybrid)
    """
    if not track_name and not track_id:
        raise HTTPException(status_code=400, detail="Either track_name or track_id must be provided")
    
    found_track_name = None
    
    # If track ID is not provided, look it up by name
    if not track_id and track_name:
        track_id, found_track_name = get_track_id_by_name(track_name)
    else:
        # If track_id is provided, get the track name
        if df is not None and track_id in df['track_id'].values:
            found_track_name = df[df['track_id'] == track_id]['track_name'].iloc[0]
        else:
            raise HTTPException(status_code=404, detail=f"Track ID '{track_id}' not found")
    
    # Generate recommendations based on model type
    # Currently we only have hybrid model implemented, but this allows for future expansion
    results = hybrid_recommend(track_id, k)
            
    return {
        "track_name": found_track_name or track_name,
        "track_id": track_id,
        "recommendations": results
    }

@app.get("/search", tags=["Search"])
async def search_tracks(query: str = Query(..., description="Search query for track names")):
    """
    Search for tracks by name.
    
    - **query**: Search term for finding tracks
    """
    if df is None:
        raise HTTPException(status_code=503, detail="Database not loaded. Please try again later.")
    
    # Case-insensitive search that handles NaN values
    # First, filter out any rows where track_name is NaN
    valid_tracks = df.dropna(subset=['track_name'])
    
    # Now perform the search on the clean data
    matches = valid_tracks[valid_tracks['track_name'].str.lower().str.contains(query.lower())]
    
    if matches.empty:
        return {"results": [], "count": 0}
    
    # Limit to top 10 matches
    results = matches[['track_id', 'track_name', 'artists']].head(10).to_dict(orient="records")
    return {"results": results, "count": len(matches)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)