import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, mean_absolute_error
from collections import Counter
import math

# Load datasets
tourism_data = pd.read_csv('tourism_with_id.csv')
user_data = pd.read_csv('user.csv')
rating_data = pd.read_csv('tourism_rating.csv')

# Calculate average ratings for each place
place_ratings = rating_data.groupby('Place_Id')['Place_Ratings'].mean().reset_index()
place_ratings.columns = ['Place_Id', 'Average_Rating']

# Merge ratings with tourism data
tourism_data = tourism_data.merge(place_ratings, on='Place_Id', how='left')
tourism_data['Average_Rating'] = tourism_data['Average_Rating'].fillna(tourism_data['Average_Rating'].mean())

def analyze_models():
    """Analyze and evaluate all models used in the recommendation system"""
    print("\n🧠 MODEL & ALGORITMA")
    print("\n1. K-Means Clustering (Machine Learning)")
    print("   - Digunakan untuk mengelompokkan tempat wisata berdasarkan fitur (Category, City, Price)")
    print("   - Alasan: Efisien untuk segmentasi data, mudah diinterpretasikan, membantu menemukan pola tersembunyi")
    print("   - Termasuk dalam kategori: Unsupervised Learning")
    print("   - Fungsi: Mengelompokkan tempat wisata dan memberikan cluster similarity score")
    
    print("\n2. TF-IDF + Cosine Similarity (Non-Machine Learning)")
    print("   - TF-IDF: Feature extraction untuk analisis teks")
    print("   - Cosine Similarity: Metrik matematis untuk mengukur kemiripan")
    print("   - Alasan: Efektif untuk analisis konten tanpa memerlukan pembelajaran")
    
    print("\n3. Rule-Based Scoring (Non-Machine Learning)")
    print("   - Location Score: Berdasarkan aturan kedekatan lokasi")
    print("   - Price Score: Berdasarkan aturan kesesuaian budget")
    print("   - Category Score: Berdasarkan aturan preferensi kategori")
    print("   - Rating Score: Berdasarkan normalisasi rating")
    
    print("\n⚙️ TRAINING PROCESS")
    print("\n1. K-Means (Machine Learning):")
    print("   - n_clusters = 5 (default)")
    print("   - random_state = 42")
    print("   - max_iter = 300")
    print("   - Proses: Mengelompokkan data berdasarkan kemiripan fitur")
    print("   - Output: Cluster assignment untuk setiap tempat wisata")
    
    print("\n2. TF-IDF (Feature Extraction):")
    print("   - stop_words = 'english'")
    print("   - max_features = None (gunakan semua kata)")
    print("   - Proses: Mengubah teks menjadi vektor numerik")
    
    print("\n📏 EVALUASI MODEL")
    X, _, _ = prepare_ml_data()
    description_matrix, tfidf = create_content_features()
    
    # K-Means Evaluation
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, clusters)
    print("\n1. K-Means (Machine Learning):")
    print(f"   - Silhouette Score: {silhouette_avg:.3f}")
    print("   - Interpretasi: Score > 0.5 menunjukkan cluster yang cukup terpisah")
    print("   - Penggunaan: Memberikan cluster similarity score dalam rekomendasi")
    
    # TF-IDF Analysis
    print("\n2. TF-IDF (Feature Extraction):")
    feature_names = tfidf.get_feature_names_out()
    avg_tfidf = description_matrix.mean(axis=0).A1
    word_tfidf = list(zip(feature_names, avg_tfidf))
    word_tfidf.sort(key=lambda x: x[1], reverse=True)
    print("   - Top 5 kata terpenting dalam deskripsi:")
    for word, score in word_tfidf[:5]:
        print(f"     * {word}: {score:.3f}")
    
    print("\n🎯 HYPERPARAMETER TUNING")
    print("\n1. K-Means (Machine Learning):")
    print("   - Mencoba n_clusters = [3, 5, 7]")
    print("   - Evaluasi menggunakan Silhouette Score")
    print("   - Hasil tuning akan ditampilkan saat menjalankan model")
    
    print("\nBOBOT MODEL DALAM REKOMENDASI")
    print("\n1. Location Score (20%) - Rule-Based")
    print("   - Berdasarkan kedekatan lokasi dengan user")
    print("   - Score tertinggi untuk tempat di provinsi yang sama")
    
    print("\n2. Price Score (20%) - Rule-Based")
    print("   - Berdasarkan kesesuaian dengan budget user")
    print("   - Score tertinggi untuk tempat dengan harga ≤ 20% budget")
    
    print("\n3. Category Score (15%) - Rule-Based")
    print("   - Berdasarkan preferensi kategori user")
    print("   - Score 1.0 untuk kategori yang sesuai, 0.5 untuk yang tidak")
    
    print("\n4. Rating Score (15%) - Rule-Based")
    print("   - Berdasarkan rata-rata rating tempat")
    print("   - Dinormalisasi ke skala 0-1")
    
    print("\n5. Content Similarity (10%) - TF-IDF + Cosine Similarity")
    print("   - Berdasarkan kemiripan deskripsi tempat")
    print("   - Menggunakan TF-IDF untuk ekstraksi fitur dan Cosine Similarity untuk perhitungan kemiripan")
    
    print("\n6. Cluster Similarity (20%) - K-Means")
    print("   - Berdasarkan kesamaan cluster antara user dan tempat")
    print("   - Score 1.0 untuk tempat dalam cluster yang sama, 0.5 untuk cluster berbeda")

# Prepare data for ML models
def prepare_ml_data():
    # Create LabelEncoders
    category_le = LabelEncoder()
    city_le = LabelEncoder()
    
    # Fit encoders with all possible categories
    all_categories = pd.concat([
        tourism_data['Category'],
        pd.Series(user_data['Preferensi'].unique())
    ]).unique()
    
    all_cities = pd.concat([
        tourism_data['City'],
        pd.Series([loc.split(', ')[0] for loc in user_data['Location'].unique()])
    ]).unique()
    
    category_le.fit(all_categories)
    city_le.fit(all_cities)
    
    # Transform categories
    tourism_data['Category_Encoded'] = category_le.transform(tourism_data['Category'])
    tourism_data['City_Encoded'] = city_le.transform(tourism_data['City'])
    
    # Create features for ML
    X = tourism_data[['Category_Encoded', 'City_Encoded', 'Price']]
    
    return X, category_le, city_le

def create_content_features():
    """Create content-based features using TF-IDF on descriptions with enhanced text processing"""
    # Combine relevant text features for better content analysis
    tourism_data['combined_text'] = tourism_data.apply(
        lambda x: f"{x['Description']} {x['Category']} {x['City']}", 
        axis=1
    )
    
    # Configure TF-IDF with better parameters for tourism content
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=1000,  # Limit features to most important terms
        ngram_range=(1, 2),  # Include both unigrams and bigrams
        min_df=2,  # Minimum document frequency
        max_df=0.95  # Maximum document frequency
    )
    
    # Transform the combined text
    description_matrix = tfidf.fit_transform(tourism_data['combined_text'].fillna(''))
    
    return description_matrix, tfidf

# Perform K-means clustering on places
def cluster_places():
    X, _, _ = prepare_ml_data()
    kmeans = KMeans(n_clusters=5, random_state=42)
    tourism_data['Cluster'] = kmeans.fit_predict(X)
    return kmeans

def get_user_preferences(user_id):
    """Get user preferences based on their location, age, budget, and category preference"""
    try:
        user_info = user_data[user_data['User_Id'] == user_id].iloc[0]
        return {
            'location': user_info['Location'],
            'age': user_info['Age'],
            'budget': user_info['Budget'],
            'category_preference': user_info['Preferensi']
        }
    except IndexError:
        # If user not found, return None
        return None

def calculate_location_score(place_city, user_location, place_lat, place_lng, user_lat=None, user_lng=None):
    """Calculate score based on location proximity using coordinates"""
    # Extract city and province from user location
    user_city, user_province = user_location.split(', ')
    place_province = place_city.split(', ')[-1] if ', ' in place_city else place_city
    
    # Base score on province match
    if user_province == place_province:
        base_score = 1.0
    elif user_province in ['Jawa Barat', 'Jawa Tengah', 'Jawa Timur', 'DIY'] and \
         place_province in ['Jawa Barat', 'Jawa Tengah', 'Jawa Timur', 'DIY']:
        base_score = 0.8
    else:
        base_score = 0.5
    
    # If we have coordinates for both places, calculate distance score
    if user_lat is not None and user_lng is not None and place_lat is not None and place_lng is not None:
        # Calculate distance using Haversine formula
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1 = float(user_lat), float(user_lng)
        lat2, lon2 = float(place_lat), float(place_lng)
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = math.sin(dlat/2) * math.sin(dlat/2) + \
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
            math.sin(dlon/2) * math.sin(dlon/2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        # Convert distance to score (closer = higher score)
        distance_score = 1.0 / (1.0 + distance/100)  # Normalize to 0-1 range
        
        # Combine base score with distance score
        return 0.7 * base_score + 0.3 * distance_score
    
    return base_score

def calculate_price_score(place_price, user_budget):
    """Calculate score based on user's budget"""
    if user_budget == 0:
        return 1.0 if place_price == 0 else 0.0
    
    price_ratio = place_price / user_budget
    
    if price_ratio <= 0.2:
        return 1.0
    elif price_ratio <= 0.5:
        return 0.8
    elif price_ratio <= 0.8:
        return 0.6
    else:
        return 0.3

def calculate_category_score(place_category, user_preference):
    """Calculate score based on category preference"""
    if place_category == user_preference:
        return 1.0
    else:
        return 0.5

def calculate_rating_score(place_rating):
    """Calculate score based on place rating"""
    return place_rating / 5.0  # Normalize rating to 0-1 scale

def get_recommendations(user_preferences, n_recommendations=5):
    """
    Get recommendations based on user preferences using content-based filtering
    with improved location handling
    """
    # Calculate place popularity
    popularity = calculate_place_popularity()
    
    # Prepare data for ML models
    X, category_le, city_le = prepare_ml_data()
    
    # Train ML models
    kmeans = cluster_places()
    description_matrix, tfidf = create_content_features()
    
    # Get user's preferred cluster
    user_features = np.array([[
        category_le.transform([user_preferences['category_preference']])[0],
        city_le.transform([user_preferences['location'].split(', ')[0]])[0],
        user_preferences['budget']
    ]])
    user_cluster = kmeans.predict(user_features)[0]
    
    # Get user's location coordinates (if available)
    user_city = user_preferences['location'].split(', ')[0]
    user_location_data = tourism_data[tourism_data['City'] == user_city]
    user_lat = user_location_data['Lat'].iloc[0] if not user_location_data.empty else None
    user_lng = user_location_data['Long'].iloc[0] if not user_location_data.empty else None
    
    # Calculate scores for each place
    scores = []
    for _, place in tourism_data.iterrows():
        # Location score with coordinates
        location_score = calculate_location_score(
            place['City'],
            user_preferences['location'],
            place['Lat'],
            place['Long'],
            user_lat,
            user_lng
        )
        
        # Price score (higher weight for new users)
        price_score = calculate_price_score(place['Price'], user_preferences['budget'])
        
        # Category score (higher weight for new users)
        category_score = calculate_category_score(place['Category'], user_preferences['category_preference'])
        
        # Content-based similarity score
        place_description = tfidf.transform([place['combined_text']])
        content_similarity = cosine_similarity(place_description, description_matrix).mean()
        
        # Cluster similarity score
        place_features = np.array([[
            category_le.transform([place['Category']])[0],
            city_le.transform([place['City']])[0],
            place['Price']
        ]])
        place_cluster = kmeans.predict(place_features)[0]
        cluster_score = 1.0 if place_cluster == user_cluster else 0.5
        
        # Popularity score
        popularity_score = popularity[
            popularity['Place_Id'] == place['Place_Id']
        ]['Popularity_Score'].values[0] if place['Place_Id'] in popularity['Place_Id'].values else 0.5
        
        # Calculate final score with adjusted weights
        final_score = (
            0.35 * location_score +      # Location importance (increased)
            0.25 * price_score +        # Budget importance
            0.20 * category_score +     # Category preference
            0.10 * content_similarity + # Content-based similarity
            0.10 * popularity_score     # Popularity score
        )
        
        scores.append({
            'Place_Id': place['Place_Id'],
            'Place_Name': place['Place_Name'],
            'Category': place['Category'],
            'City': place['City'],
            'Price': place['Price'],
            'Content_Score': content_similarity,
            'Popularity_Score': popularity_score,
            'Location_Score': location_score,
            'Score': final_score
        })
    
    # Convert to DataFrame and sort by score
    recommendations_df = pd.DataFrame(scores)
    
    # Add diversity to recommendations
    # Group by city and take top 2 from each city
    top_by_city = recommendations_df.sort_values('Score', ascending=False).groupby('City').head(2)
    
    # Get remaining recommendations
    remaining = recommendations_df[~recommendations_df.index.isin(top_by_city.index)]
    remaining = remaining.sort_values('Score', ascending=False).head(n_recommendations - len(top_by_city))
    
    # Combine and sort final recommendations
    final_recommendations = pd.concat([top_by_city, remaining])
    final_recommendations = final_recommendations.sort_values('Score', ascending=False)
    
    return final_recommendations.head(n_recommendations).to_dict('records')

def evaluate_models():
    """Evaluate KMeans clustering"""
    X, _, _ = prepare_ml_data()
    
    # K-Means Evaluation
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, clusters)
    print(f"K-Means Silhouette Score: {silhouette_avg:.3f}")
    
    return kmeans

def tune_kmeans_manual(X, cluster_range=[3, 5, 7]):
    best_score = -1
    best_k = None
    best_model = None
    for k in cluster_range:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"n_clusters={k}, silhouette_score={score:.3f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model
    print(f"Best n_clusters: {best_k} with silhouette_score: {best_score:.3f}")
    return best_model

def tune_hyperparameters():
    """Tune hyperparameters for KMeans"""
    X, _, _ = prepare_ml_data()
    
    # K-Means Tuning (manual)
    print("\nManual tuning for KMeans:")
    best_kmeans = tune_kmeans_manual(X, cluster_range=[3, 5, 7])
    
    return best_kmeans

def get_new_user_recommendations(user_id, n_recommendations=5):
    """
    Get recommendations specifically for new users, focusing on content-based features
    and user preferences without relying on ratings.
    """
    # Get user preferences
    preferences = get_user_preferences(user_id)
    
    # Prepare data for ML models
    X, category_le, city_le = prepare_ml_data()
    
    # Train ML models
    kmeans = cluster_places()
    description_matrix, tfidf = create_content_features()
    
    # Get user's preferred cluster based on their preferences
    user_features = np.array([[
        category_le.transform([preferences['category_preference']])[0],
        city_le.transform([preferences['location'].split(', ')[0]])[0],
        preferences['budget']
    ]])
    user_cluster = kmeans.predict(user_features)[0]
    
    # Calculate scores for each place
    scores = []
    for _, place in tourism_data.iterrows():
        # Location score (higher weight for new users)
        location_score = calculate_location_score(
            place['City'],
            preferences['location'],
            place['Lat'],
            place['Long'],
            None,
            None
        )
        
        # Price score (higher weight for new users)
        price_score = calculate_price_score(place['Price'], preferences['budget'])
        
        # Category score (higher weight for new users)
        category_score = calculate_category_score(place['Category'], preferences['category_preference'])
        
        # Content-based similarity score
        place_description = tfidf.transform([place['combined_text']])
        content_similarity = cosine_similarity(place_description, description_matrix).mean()
        
        # Cluster similarity score
        place_features = np.array([[
            category_le.transform([place['Category']])[0],
            city_le.transform([place['City']])[0],
            place['Price']
        ]])
        place_cluster = kmeans.predict(place_features)[0]
        cluster_score = 1.0 if place_cluster == user_cluster else 0.5
        
        # Calculate final score with adjusted weights for new users
        final_score = (
            0.30 * location_score +      # Location importance (increased)
            0.25 * price_score +        # Budget importance (increased)
            0.20 * category_score +     # Category preference (increased)
            0.15 * content_similarity + # Content-based similarity (increased)
            0.10 * cluster_score        # Cluster similarity (reduced)
        )
        
        scores.append({
            'Place_Id': place['Place_Id'],
            'Place_Name': place['Place_Name'],
            'Category': place['Category'],
            'City': place['City'],
            'Price': place['Price'],
            'Content_Score': content_similarity,
            'Cluster_Score': cluster_score,
            'Score': final_score
        })
    
    # Convert to DataFrame and sort by score
    recommendations_df = pd.DataFrame(scores)
    recommendations_df = recommendations_df.sort_values('Score', ascending=False)
    
    return recommendations_df.head(n_recommendations).to_dict('records')

def calculate_place_popularity():
    """Calculate popularity score for each place based on ratings and other factors"""
    # Calculate average rating and number of ratings for each place
    popularity = rating_data.groupby('Place_Id').agg({
        'Place_Ratings': ['mean', 'count']
    }).reset_index()
    
    # Rename columns for clarity
    popularity.columns = ['Place_Id', 'Avg_Rating', 'Rating_Count']
    
    # Normalize rating count to 0-1 scale
    popularity['Rating_Count_Normalized'] = popularity['Rating_Count'] / popularity['Rating_Count'].max()
    
    # Calculate popularity score (weighted combination of average rating and rating count)
    popularity['Popularity_Score'] = (
        0.6 * popularity['Avg_Rating'] / 5.0 +  # Normalize rating to 0-1 scale
        0.4 * popularity['Rating_Count_Normalized']
    )
    
    # Fill missing values with median score
    median_score = popularity['Popularity_Score'].median()
    popularity['Popularity_Score'] = popularity['Popularity_Score'].fillna(median_score)
    
    return popularity

def get_user_rating_history(user_id):
    """Get rating history and preferences for a specific user"""
    user_ratings = rating_data[rating_data['User_Id'] == user_id]
    
    if len(user_ratings) == 0:
        return None
    
    # Get places rated by user
    rated_places = tourism_data[tourism_data['Place_Id'].isin(user_ratings['Place_Id'])]
    
    # Calculate user preferences
    preferences = {
        'categories': rated_places['Category'].value_counts(normalize=True).to_dict(),
        'cities': rated_places['City'].value_counts(normalize=True).to_dict(),
        'avg_price': rated_places['Price'].mean(),
        'price_range': (rated_places['Price'].min(), rated_places['Price'].max())
    }
    
    return preferences

def calculate_rating_based_score(place_id, user_id):
    """Calculate score based on rating patterns"""
    # Get user's rating history
    user_ratings = rating_data[rating_data['User_Id'] == user_id]
    
    if len(user_ratings) == 0:
        return 0.5  # Default score for new users
    
    # Get places with similar ratings
    similar_ratings = rating_data[
        (rating_data['User_Id'].isin(user_ratings['User_Id'])) &
        (rating_data['Place_Ratings'] >= 4)  # Consider only highly rated places
    ]
    
    # Calculate similarity score based on rating patterns
    if place_id in similar_ratings['Place_Id'].values:
        return 1.0
    else:
        return 0.5

def get_hybrid_recommendations(user_id, n_recommendations=5, is_new_user=False):
    """
    Get recommendations using a hybrid approach that combines content-based filtering
    with rating-based insights
    """
    # Get user preferences
    preferences = get_user_preferences(user_id)
    
    if preferences is None:
        # If user not found, use default preferences from available data
        default_category = user_data['Preferensi'].iloc[0]  # Get first available category
        default_location = user_data['Location'].iloc[0]    # Get first available location
        
        preferences = {
            'location': default_location,
            'age': 25,  # Default age
            'budget': 1000000,  # Default budget
            'category_preference': default_category
        }
    
    # Calculate place popularity
    popularity = calculate_place_popularity()
    
    # Get user rating history
    user_history = get_user_rating_history(user_id)
    
    # Prepare data for ML models
    X, category_le, city_le = prepare_ml_data()
    
    # Train ML models
    kmeans = cluster_places()
    description_matrix, tfidf = create_content_features()
    
    # Get user's preferred cluster
    user_features = np.array([[
        category_le.transform([preferences['category_preference']])[0],
        city_le.transform([preferences['location'].split(', ')[0]])[0],
        preferences['budget']
    ]])
    user_cluster = kmeans.predict(user_features)[0]
    
    # Calculate scores for each place
    scores = []
    for _, place in tourism_data.iterrows():
        # Location score
        location_score = calculate_location_score(
            place['City'],
            preferences['location'],
            place['Lat'],
            place['Long'],
            None,
            None
        )
        
        # Price score
        price_score = calculate_price_score(place['Price'], preferences['budget'])
        
        # Category score
        category_score = calculate_category_score(place['Category'], preferences['category_preference'])
        
        # Content-based similarity score
        place_description = tfidf.transform([place['combined_text']])
        content_similarity = cosine_similarity(place_description, description_matrix).mean()
        
        # Cluster similarity score
        place_features = np.array([[
            category_le.transform([place['Category']])[0],
            city_le.transform([place['City']])[0],
            place['Price']
        ]])
        place_cluster = kmeans.predict(place_features)[0]
        cluster_score = 1.0 if place_cluster == user_cluster else 0.5
        
        # Rating-based score
        rating_score = calculate_rating_based_score(place['Place_Id'], user_id)
        
        # Popularity score
        popularity_score = popularity[
            popularity['Place_Id'] == place['Place_Id']
        ]['Popularity_Score'].values[0] if place['Place_Id'] in popularity['Place_Id'].values else 0.5
        
        # Calculate final score with adjusted weights
        if is_new_user:
            final_score = (
                0.25 * location_score +      # Location importance
                0.20 * price_score +        # Budget importance
                0.15 * category_score +     # Category preference
                0.15 * content_similarity + # Content-based similarity
                0.10 * cluster_score +      # Cluster similarity
                0.15 * popularity_score     # Popularity score
            )
        else:
            final_score = (
                0.20 * location_score +      # Location importance
                0.15 * price_score +        # Budget importance
                0.15 * category_score +     # Category preference
                0.10 * content_similarity + # Content-based similarity
                0.10 * cluster_score +      # Cluster similarity
                0.15 * rating_score +       # Rating-based score
                0.15 * popularity_score     # Popularity score
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
    
    # Convert to DataFrame and sort by score
    recommendations_df = pd.DataFrame(scores)
    recommendations_df = recommendations_df.sort_values('Score', ascending=False)
    
    return recommendations_df.head(n_recommendations).to_dict('records')

# Main execution
if __name__ == "__main__":
    # Analyze models first
    analyze_models()
    
    print("\nEvaluating models...")
    kmeans = evaluate_models()
    
    print("\nTuning hyperparameters...")
    best_kmeans = tune_hyperparameters()
    
    # Get recommendations using tuned models
    user_id = 1
    recommendations = get_recommendations(get_user_preferences(user_id))
    
    # Get user preferences for display
    user_prefs = get_user_preferences(user_id)
    
    # Print user preferences
    print("\nUser Preferences:")
    print(f"Location: {user_prefs['location']}")
    print(f"Age: {user_prefs['age']}")
    print(f"Budget: {user_prefs['budget']}")
    print(f"Category Preference: {user_prefs['category_preference']}")
    
    # Print recommendations
    print("\nRecommended Places:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['Place_Name']}")
        print(f"   Category: {rec['Category']}")
        print(f"   City: {rec['City']}")
        print(f"   Price: {rec['Price']}")
        print(f"   Average Rating: {rec['Average_Rating']:.1f}")
        print(f"   Content Score: {rec['Content_Score']:.2f}")
        print(f"   Cluster Score: {rec['Cluster_Score']:.2f}")
        print(f"   Recommendation Score: {rec['Score']:.2f}") 