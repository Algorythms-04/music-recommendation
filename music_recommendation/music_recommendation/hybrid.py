import pickle
import faiss
import numpy as np
import pandas as pd

# ✅ 1. Load the SVD model
with open("svd_model.pkl", "rb") as f:
    svd_model = pickle.load(f)
print("✅ SVD model loaded.")

# ✅ 2. Load the FAISS index
index = faiss.read_index("faiss_index.bin")
print("✅ FAISS index loaded.")

# ✅ 3. Load the track ID mapping
df = pd.read_csv("track_id_mapping.csv", index_col=0)
print("✅ Track ID mapping loaded.")

# ✅ 4. Load the feature matrix for FAISS search
feature_matrix = np.load("feature_matrix.npy")
print("✅ Feature matrix loaded.")


import pandas as pd

def hybrid_recommend(track_id, k=5):
    """
    Generates recommendations using FAISS (content-based filtering).

    Args:
        track_id (str): The track ID for which recommendations are needed.
        k (int, optional): Number of recommendations to return. Defaults to 5.

    Returns:
        list: A list of recommended song names and artists in JSON format.
    """

    # ✅ Step 1: Check if track_id exists
    if track_id not in df['track_id'].values:
        return {"error": f"⚠️ Track ID '{track_id}' not found!"}

    # ✅ Step 2: Get track index for FAISS
    track_idx = df.index[df['track_id'] == track_id].tolist()[0]

    # ✅ Step 3: Get similar tracks from FAISS
    query_vector = feature_matrix[track_idx].reshape(1, -1)
    distances, indices = index.search(query_vector, k+1)  # +1 to exclude itself
    similar_songs = df.iloc[indices[0][1:]]  # Exclude original song

    # ✅ Step 4: Format response as a list of dictionaries
    recommendations = similar_songs[['track_name', 'artists']].to_dict(orient="records")

    return recommendations
