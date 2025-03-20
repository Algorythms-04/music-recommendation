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
        print(f"FAISS index dimension: {index.d}")
        
        # Load the track ID mapping
        print("Loading track ID mapping...")
        df = pd.read_csv("track_id_mapping.csv", index_col=0)
        print("✅ Track ID mapping loaded.")
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame index name: {df.index.name}")
        
        # Load the feature matrix for FAISS search
        print("Loading feature matrix...")
        feature_matrix = np.load("feature_matrix.npy")
        print(f"✅ Feature matrix loaded. Shape: {feature_matrix.shape}")
        
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
    
    # Print the column names for debugging
    print(f"Searching for track: {track_name}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    
    # Case-insensitive search
    matches = df[df['track_name'].str.lower() == track_name.lower()]
    
    # If no exact match, try partial match
    if matches.empty:
        matches = df[df['track_name'].str.lower().str.contains(track_name.lower())]
    
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"No tracks found with name '{track_name}'")
    
    # Check if track_id is in the index or columns
    if 'track_id' not in df.columns:
        # If track_id is the index, use the index value
        if df.index.name == 'track_id':
            return matches.index[0], matches['track_name'].iloc[0]
        else:
            # Try to find a suitable column that might be the track ID
            id_columns = [col for col in df.columns if 'id' in col.lower()]
            if id_columns:
                return matches[id_columns[0]].iloc[0], matches['track_name'].iloc[0]
            else:
                raise HTTPException(status_code=500, detail=f"Cannot find track_id column or suitable ID column")
    else:
        # Take the first match as normal
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
    # First, determine if track_id is in the index or columns
    track_id_in_index = df.index.name == 'track_id'
    track_exists = False
    
    if track_id_in_index:
        track_exists = track_id in df.index
        print(f"Checking track_id {track_id} in index: {track_exists}")
    elif 'track_id' in df.columns:
        track_exists = track_id in df['track_id'].values
        print(f"Checking track_id {track_id} in column: {track_exists}")
    else:
        # Try to find a suitable column that might be the track ID
        id_columns = [col for col in df.columns if 'id' in col.lower()]
        if id_columns:
            track_exists = any(track_id in df[col].values for col in id_columns if pd.api.types.is_hashable(df[col].iloc[0]))
            print(f"Checking track_id {track_id} in alternative ID columns: {track_exists}")
    
    if not track_exists:
        raise HTTPException(status_code=404, detail=f"⚠️ Track ID '{track_id}' not found!")

    try:
        # Get track index for FAISS
        track_idx = None
        original_track_name = None
        
        if track_id_in_index:
            track_idx = df.index.get_loc(track_id)
            original_track_name = df.loc[track_id, 'track_name']
            # Ensure original_track_name is a string, not a Series
            if isinstance(original_track_name, pd.Series):
                original_track_name = original_track_name.iloc[0]
        elif 'track_id' in df.columns:
            track_idx = df.index[df['track_id'] == track_id].tolist()[0]
            original_track_name = df.loc[track_idx, 'track_name']
            # Ensure original_track_name is a string, not a Series
            if isinstance(original_track_name, pd.Series):
                original_track_name = original_track_name.iloc[0]
        else:
            # Try to use alternative ID columns
            id_columns = [col for col in df.columns if 'id' in col.lower()]
            for col in id_columns:
                if pd.api.types.is_hashable(df[col].iloc[0]) and track_id in df[col].values:
                    track_idx = df.index[df[col] == track_id].tolist()[0]
                    original_track_name = df.loc[track_idx, 'track_name']
                    # Ensure original_track_name is a string, not a Series
                    if isinstance(original_track_name, pd.Series):
                        original_track_name = original_track_name.iloc[0]
                    break

        if track_idx is None:
            raise HTTPException(status_code=500, detail=f"Unable to find track index for track_id '{track_id}'")

        # For debugging
        print(f"Track index: {track_idx}")
        print(f"Feature matrix shape: {feature_matrix.shape}")
        print(f"Index dimension: {index.d}")
        print(f"Track vector shape before reshape: {feature_matrix[track_idx].shape}")
        
        # Get similar tracks from FAISS - we'll need to fetch more than k to account for duplicates
        # Start with k*3 to have enough candidates after filtering
        fetch_count = k * 3
        
        # Ensure correct format for FAISS query
        query_vector = feature_matrix[track_idx].astype(np.float32)
        if len(query_vector.shape) == 1:
            # If it's a 1D array, reshape to 2D (1, d)
            query_vector = query_vector.reshape(1, -1)
        
        print(f"Query vector shape after reshape: {query_vector.shape}")
        
        # Verify dimensions match
        if query_vector.shape[1] != index.d:
            error_msg = f"Dimension mismatch: Query vector has {query_vector.shape[1]} dimensions, index expects {index.d}"
            print(f"⚠️ {error_msg}")
            
            # Try to adapt the vector if possible (e.g., by truncating or padding)
            if query_vector.shape[1] > index.d:
                print(f"Truncating query vector from {query_vector.shape[1]} to {index.d} dimensions")
                query_vector = query_vector[:, :index.d]
            else:
                # If we can't adapt, raise an exception
                raise HTTPException(status_code=500, detail=error_msg)
        
        # Get recommendations until we have enough unique ones
        unique_recommendations = []
        seen_track_names = set([original_track_name])  # Start with original track to exclude it
        
        while len(unique_recommendations) < k:
            distances, indices = index.search(query_vector, fetch_count)
            
            # Safety check for indices
            if len(indices[0]) == 0:
                print("⚠️ FAISS returned empty indices")
                break
                
            # Check if indices are out of bounds
            valid_indices = [idx for idx in indices[0] if 0 <= idx < len(df)]
            if len(valid_indices) == 0:
                print("⚠️ FAISS returned only out-of-bounds indices")
                break
                
            # Use only valid indices
            similar_songs = df.iloc[valid_indices[1:]]  # Exclude original song
            
            # Filter for unique track names
            for _, song in similar_songs.iterrows():
                track_name = song['track_name']
                # Ensure track_name is a string, not a Series
                if isinstance(track_name, pd.Series):
                    track_name = track_name.iloc[0]
                    
                artists = song['artists']
                # Ensure artists is a string, not a Series
                if isinstance(artists, pd.Series):
                    artists = artists.iloc[0]
                    
                if track_name not in seen_track_names:
                    unique_recommendations.append({
                        'track_name': track_name, 
                        'artists': artists
                    })
                    seen_track_names.add(track_name)
                    
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
        print(f"Error in hybrid_recommend: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    models_loaded = all(x is not None for x in [svd_model, index, df, feature_matrix])
    return {
        "status": "healthy" if models_loaded else "degraded",
        "message": "API is running with all models loaded" if models_loaded else "API is running but models failed to load"
    }

@app.get("/debug/dataframe", tags=["Debug"])
async def debug_dataframe():
    """Debug endpoint to check dataframe structure."""
    if df is None:
        return {"error": "DataFrame not loaded"}
    
    # Get sample data without converting NaN to null (which can cause JSON serialization issues)
    sample_data = df.head(3).fillna("NA").to_dict(orient="records")
    
    # Get column data types as strings
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    
    return {
        "columns": df.columns.tolist(),
        "index_name": df.index.name,
        "sample_data": sample_data,
        "shape": df.shape,
        "dtypes": dtypes
    }

@app.get("/debug/model-info", tags=["Debug"])
async def debug_model_info():
    """Debug endpoint to check model dimensions and compatibility."""
    if index is None or feature_matrix is None:
        return {"error": "Models not loaded"}
    
    return {
        "faiss_index_dimension": index.d,
        "faiss_index_ntotal": index.ntotal,
        "feature_matrix_shape": feature_matrix.shape,
        "feature_matrix_dtype": str(feature_matrix.dtype),
        "compatibility": {
            "dimensions_match": feature_matrix.shape[1] == index.d,
            "count_match": feature_matrix.shape[0] == df.shape[0] if df is not None else False
        }
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
        track_id_in_index = df.index.name == 'track_id'
        
        if track_id_in_index:
            if track_id in df.index:
                found_track_name = df.loc[track_id, 'track_name']
                # Ensure found_track_name is a string, not a Series
                if isinstance(found_track_name, pd.Series):
                    found_track_name = found_track_name.iloc[0]
            else:
                raise HTTPException(status_code=404, detail=f"Track ID '{track_id}' not found")
        elif 'track_id' in df.columns:
            if track_id in df['track_id'].values:
                found_track_name = df[df['track_id'] == track_id]['track_name'].iloc[0]
                # Ensure found_track_name is a string, not a Series
                if isinstance(found_track_name, pd.Series):
                    found_track_name = found_track_name.iloc[0]
            else:
                raise HTTPException(status_code=404, detail=f"Track ID '{track_id}' not found")
        else:
            # Try to find track_id in other ID columns
            id_columns = [col for col in df.columns if 'id' in col.lower()]
            found = False
            
            for col in id_columns:
                if pd.api.types.is_hashable(df[col].iloc[0]) and track_id in df[col].values:
                    found_track_name = df[df[col] == track_id]['track_name'].iloc[0]
                    # Ensure found_track_name is a string, not a Series
                    if isinstance(found_track_name, pd.Series):
                        found_track_name = found_track_name.iloc[0]
                    found = True
                    break
                    
            if not found:
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
    
    # Prepare results with ID field
    results = []
    for _, row in matches.head(10).iterrows():
        track_name = row['track_name']
        # Ensure track_name is a string, not a Series
        if isinstance(track_name, pd.Series):
            track_name = track_name.iloc[0]
            
        artists = row['artists']
        # Ensure artists is a string, not a Series
        if isinstance(artists, pd.Series):
            artists = artists.iloc[0]
            
        result = {
            'track_name': track_name, 
            'artists': artists
        }
        
        # Add track_id from either column or index
        if 'track_id' in df.columns:
            track_id = row['track_id']
            # Ensure track_id is a string or primitive, not a Series
            if isinstance(track_id, pd.Series):
                track_id = track_id.iloc[0]
            result['track_id'] = track_id
        elif df.index.name == 'track_id':
            result['track_id'] = row.name
        else:
            # Try to use first column with 'id' in the name
            id_columns = [col for col in df.columns if 'id' in col.lower()]
            if id_columns:
                track_id = row[id_columns[0]]
                # Ensure track_id is a string or primitive, not a Series
                if isinstance(track_id, pd.Series):
                    track_id = track_id.iloc[0]
                result['track_id'] = track_id
            else:
                # Fallback to index if no ID column is found
                result['track_id'] = str(row.name)
                
        results.append(result)
        
    return {"results": results, "count": len(matches)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)