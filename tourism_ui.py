import streamlit as st
import pandas as pd
from tourism_recommendation import (
    get_recommendations,
    user_data,
    tourism_data
)

# Set page configuration
st.set_page_config(
    page_title="Tourism Recommendation System",
    page_icon="🏖️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .recommendation-card {
        background-color: #1E1E1E;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #FFFFFF;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .recommendation-card h3 {
        color: #4CAF50;
        margin-bottom: 1rem;
        font-size: 1.5rem;
    }
    .recommendation-card p {
        color: #E0E0E0;
        margin: 0.5rem 0;
        font-size: 1rem;
    }
    .recommendation-card strong {
        color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🏖️ Tourism Recommendation System")
st.markdown("---")

# Sidebar for user input
st.sidebar.header("User Information")

# Get unique locations and categories for dropdowns
try:
    locations = sorted(user_data['Location'].unique())
    categories = sorted(user_data['Preferensi'].unique())
    
    if len(locations) == 0 or len(categories) == 0:
        st.error("Error: No locations or categories found in the data.")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# User input form
with st.sidebar.form("user_input_form"):
    st.subheader("Enter Your Preferences")
    
    # Location selection
    location = st.selectbox(
        "Select Your Location",
        locations,
        index=0
    )
    
    # Age input
    age = st.number_input(
        "Enter Your Age",
        min_value=1,
        max_value=100,
        value=25
    )
    
    # Budget input
    budget = st.number_input(
        "Enter Your Budget (in IDR)",
        min_value=0,
        value=1000000,
        step=100000
    )
    
    # Category preference
    category = st.selectbox(
        "Select Your Preferred Category",
        categories,
        index=0
    )
    
    # Submit button
    submit_button = st.form_submit_button("Get Recommendations")

# Main content area
if submit_button:
    try:
        # Create user preferences dictionary
        user_preferences = {
            'location': location,
            'age': age,
            'budget': budget,
            'category_preference': category
        }
        
        # Get recommendations
        recommendations = get_recommendations(user_preferences)
        
        # Display user preferences
        st.subheader("Your Preferences")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Location", location)
        with col2:
            st.metric("Age", age)
        with col3:
            st.metric("Budget", f"Rp {budget:,}".replace(",", "."))
        with col4:
            st.metric("Category", category)
        
        st.markdown("---")
        
        # Display recommendations
        st.subheader("Recommended Places")
        
        for i, rec in enumerate(recommendations, 1):
            with st.container():
                formatted_price = f"{rec['Price']:,}".replace(",", ".")
                st.markdown(f"""
                    <div class="recommendation-card">
                        <h3>{i}. {rec['Place_Name']}</h3>
                        <p><strong>Category:</strong> {rec['Category']}</p>
                        <p><strong>City:</strong> {rec['City']}</p>
                        <p><strong>Price:</strong> Rp {formatted_price}</p>
                        <p><strong>Content Score:</strong> {rec['Content_Score']:.2f}</p>
                        <p><strong>Popularity Score:</strong> {rec['Popularity_Score']:.2f}</p>
                        <p><strong>Final Score:</strong> {rec['Score']:.2f}</p>
                    </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try again with different preferences.")

# Add information about the system
st.sidebar.markdown("---")
st.sidebar.info("""
    ### About the System
    This recommendation system uses:
    - Content-based filtering
    - User preferences
    - Location-based recommendations
    - Popularity scores
""")

# Add instructions
st.sidebar.markdown("---")
st.sidebar.info("""
    ### How to Use
    1. Fill in your preferences
    2. Click "Get Recommendations"
    3. View your personalized recommendations
""") 