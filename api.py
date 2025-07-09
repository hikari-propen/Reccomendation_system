from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# Import logic from your script
from tourism_recommendation import (
    prepare_ml_data,
    cluster_places,
    create_content_features,
    get_user_preferences,
    calculate_location_score,
    calculate_price_score,
    calculate_category_score,
    calculate_rating_based_score,
    calculate_place_popularity,
    tourism_data,
    user_data,
    rating_data
)

app = FastAPI(
    title="Tourism Recommendation API",
    description="API for providing tourism recommendations based on user preferences.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan ["http://localhost:5173"] untuk lebih aman
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Global objects ---
# These objects are loaded once at startup to improve performance.
print("Loading data and training models... This may take a moment.")
X, category_le, city_le = prepare_ml_data()
kmeans = cluster_places()
description_matrix, tfidf = create_content_features()
popularity = calculate_place_popularity()
# Add combined_text to tourism_data if it's not already there from create_content_features
if 'combined_text' not in tourism_data.columns:
    tourism_data['combined_text'] = tourism_data.apply(
        lambda x: f"{x['Description']} {x['Category']} {x['City']}", 
        axis=1
    )
print("✅ API is ready to accept requests.")


# --- Pydantic Models for Response ---
class Recommendation(BaseModel):
    Place_Id: int
    Place_Name: str
    Category: str
    City: str
    Price: int
    Content_Score: float
    Cluster_Score: float
    Rating_Score: float
    Popularity_Score: float
    Score: float

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[Recommendation]

class CustomUserRequest(BaseModel):
    location: str
    age: int
    budget: int
    category_preference: str
    lat: Optional[float] = None
    lng: Optional[float] = None

@app.get("/")
def read_root():
    """ A simple endpoint to test if the API is running. """
    return {"message": "Welcome to the Tourism Recommendation API. Go to /docs for documentation."}

@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def get_recommendations_endpoint(user_id: int, n_recommendations: int = 5):
    """
    Generates tourism recommendations for a given user ID using a hybrid model.
    """
    preferences = get_user_preferences(user_id)
    is_new_user = False

    if preferences is None:
        # If user not found, use default preferences and mark as new user
        is_new_user = True
        default_category = user_data['Preferensi'].mode()[0]
        default_location = user_data['Location'].mode()[0]
        preferences = {
            'location': default_location,
            'age': user_data['Age'].median(),
            'budget': user_data['Budget'].median(),
            'category_preference': default_category
        }

    # Get user's preferred cluster
    try:
        user_cat_encoded = category_le.transform([preferences['category_preference']])[0]
        user_city_encoded = city_le.transform([preferences['location'].split(', ')[0]])[0]
    except ValueError:
        # Handle cases where user preference is not in the training data of the encoder
        raise HTTPException(status_code=404, detail=f"Category or City for user {user_id} not found in model vocabulary.")

    user_features = np.array([[
        user_cat_encoded,
        user_city_encoded,
        preferences['budget']
    ]])
    user_cluster = kmeans.predict(user_features)[0]

    # Calculate scores for each place
    scores = []
    for _, place in tourism_data.iterrows():
        location_score = calculate_location_score(place['City'], preferences['location'], place['Lat'], place['Long'])
        price_score = calculate_price_score(place['Price'], preferences['budget'])
        category_score = calculate_category_score(place['Category'], preferences['category_preference'])
        
        place_description = tfidf.transform([place['combined_text']])
        content_similarity = cosine_similarity(place_description, description_matrix).mean()
        
        place_cluster = place['Cluster']
        cluster_score = 1.0 if place_cluster == user_cluster else 0.5
        
        rating_score = calculate_rating_based_score(place['Place_Id'], user_id)
        
        popularity_score = popularity.loc[popularity['Place_Id'] == place['Place_Id'], 'Popularity_Score'].values[0] if place['Place_Id'] in popularity['Place_Id'].values else 0.5
        
        # Adjust weights based on whether the user is new or existing
        if is_new_user:
            final_score = (
                0.25 * location_score + 0.20 * price_score + 0.15 * category_score +
                0.15 * content_similarity + 0.10 * cluster_score + 0.15 * popularity_score
            )
        else:
            final_score = (
                0.20 * location_score + 0.15 * price_score + 0.15 * category_score +
                0.10 * content_similarity + 0.10 * cluster_score + 0.15 * rating_score + 0.15 * popularity_score
            )
        
        scores.append({
            'Place_Id': place['Place_Id'],
            'Place_Name': place['Place_Name'],
            'Category': place['Category'],
            'City': place['City'],
            'Price': place['Price'],
            'Content_Score': content_similarity,
            'Cluster_Score': cluster_score,
            'Rating_Score': rating_score,
            'Popularity_Score': popularity_score,
            'Score': final_score
        })

    recommendations_df = pd.DataFrame(scores).sort_values('Score', ascending=False)
    top_recommendations = recommendations_df.head(n_recommendations).to_dict('records')

    return {"user_id": user_id, "recommendations": top_recommendations}

@app.post("/recommendations/custom", response_model=RecommendationResponse)
def get_custom_recommendations(
    user: CustomUserRequest = Body(...),
    n_recommendations: int = 5
):
    preferences = {
        'location': user.location,
        'age': user.age,
        'budget': user.budget,
        'category_preference': user.category_preference,
        'lat': user.lat,
        'lng': user.lng
    }
    is_new_user = True  # Anggap custom user selalu new user

    # Cari lat/lng user dari tourism_data jika tidak diberikan
    if preferences['lat'] is None or preferences['lng'] is None:
        user_city = preferences['location'].split(',')[0].strip()
        main_city = user_city.split()[0] if "Jakarta" in user_city else user_city
        user_location_data = tourism_data[tourism_data['City'].str.contains(main_city, case=False, na=False)]
        if not user_location_data.empty:
            preferences['lat'] = user_location_data['Lat'].iloc[0]
            preferences['lng'] = user_location_data['Long'].iloc[0]

    # Get user's preferred cluster
    try:
        user_cat_encoded = category_le.transform([preferences['category_preference']])[0]
        user_city_encoded = city_le.transform([preferences['location'].split(', ')[0]])[0]
    except ValueError:
        raise HTTPException(status_code=404, detail="Category or City not found in model vocabulary.")

    user_features = np.array([
        [
            user_cat_encoded,
            user_city_encoded,
            preferences['budget']
        ]
    ])
    user_cluster = kmeans.predict(user_features)[0]

    # Calculate scores for each place
    scores = []
    for _, place in tourism_data.iterrows():
        location_score = calculate_location_score(
            place['City'], preferences['location'], place['Lat'], place['Long'],
            preferences['lat'], preferences['lng']
        )
        price_score = calculate_price_score(place['Price'], preferences['budget'])
        category_score = calculate_category_score(place['Category'], preferences['category_preference'])
        place_description = tfidf.transform([place['combined_text']])
        content_similarity = cosine_similarity(place_description, description_matrix).mean()
        place_cluster = place['Cluster']
        cluster_score = 1.0 if place_cluster == user_cluster else 0.5
        rating_score = 0.5  # Untuk custom user, rating score bisa default
        popularity_score = popularity.loc[popularity['Place_Id'] == place['Place_Id'], 'Popularity_Score'].values[0] if place['Place_Id'] in popularity['Place_Id'].values else 0.5

        final_score = (
            0.25 * location_score + 0.20 * price_score + 0.15 * category_score +
            0.15 * content_similarity + 0.10 * cluster_score + 0.15 * popularity_score
        )

        scores.append({
            'Place_Id': place['Place_Id'],
            'Place_Name': place['Place_Name'],
            'Category': place['Category'],
            'City': place['City'],
            'Price': place['Price'],
            'Content_Score': content_similarity,
            'Cluster_Score': cluster_score,
            'Rating_Score': rating_score,
            'Popularity_Score': popularity_score,
            'Score': final_score
        })

    recommendations_df = pd.DataFrame(scores).sort_values('Score', ascending=False)
    top_recommendations = recommendations_df.head(n_recommendations).to_dict('records')

    return {"user_id": -1, "recommendations": top_recommendations}
