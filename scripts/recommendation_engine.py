import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from pathlib import Path

class BookRecommendationEngine:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.books_df = None
        self.tags_df = None
        self.book_tags_df = None
        self.ratings_df = None
        self.to_read_df = None
        self.pivot = None
        self.pivot_tfidf = None
        self.pivot_index_map = None
        self.kmeans_models = {}
        self.data_loaded = False
    
    def load_csv_data(self):
        """Load actual CSV data files"""
        print("Loading CSV data files...")
        
        try:
            # Define file paths
            data_path = Path(self.data_dir)
            books_file = data_path / "books.csv"
            tags_file = data_path / "tags.csv"
            book_tags_file = data_path / "book_tags.csv"
            ratings_file = data_path / "ratings.csv"
            to_read_file = data_path / "to_read.csv"
            
            # Check if files exist
            required_files = [books_file, tags_file, book_tags_file]
            missing_files = [f for f in required_files if not f.exists()]
            
            if missing_files:
                print(f"Missing required files: {missing_files}")
                print("Falling back to sample data...")
                return self.load_sample_data()
            
            # Load books data
            print("Loading books.csv...")
            self.books_df = pd.read_csv(books_file)
            print(f"Loaded {len(self.books_df)} books")
            
            # Clean books data
            self.books_df = self.books_df.dropna(subset=['goodreads_book_id', 'title', 'authors'])
            self.books_df['average_rating'] = pd.to_numeric(self.books_df['average_rating'], errors='coerce').fillna(0)
            self.books_df['ratings_count'] = pd.to_numeric(self.books_df['ratings_count'], errors='coerce').fillna(0)
            
            # Handle missing image URLs
            self.books_df['image_url'] = self.books_df['image_url'].fillna('')
            
            # Load tags data
            print("Loading tags.csv...")
            self.tags_df = pd.read_csv(tags_file)
            print(f"Loaded {len(self.tags_df)} tags")
            
            # Clean tags data - filter out invalid tags
            self.tags_df = self.tags_df[
                (self.tags_df['tag_name'].notna()) & 
                (self.tags_df['tag_name'] != '') &
                (~self.tags_df['tag_name'].str.startswith('-'))  # Remove invalid tags like "--1-2"
            ]
            
            # Load book-tags relationships
            print("Loading book_tags.csv...")
            self.book_tags_df = pd.read_csv(book_tags_file)
            print(f"Loaded {len(self.book_tags_df)} book-tag relationships")
            
            # Clean book_tags data
            self.book_tags_df = self.book_tags_df.dropna(subset=['goodreads_book_id', 'tag_id', 'count'])
            self.book_tags_df = self.book_tags_df[self.book_tags_df['count'] > 0]
            
            # Filter to only include books and tags that exist in our datasets
            valid_book_ids = set(self.books_df['goodreads_book_id'])
            valid_tag_ids = set(self.tags_df['tag_id'])
            
            self.book_tags_df = self.book_tags_df[
                (self.book_tags_df['goodreads_book_id'].isin(valid_book_ids)) &
                (self.book_tags_df['tag_id'].isin(valid_tag_ids))
            ]
            
            print(f"After cleaning: {len(self.book_tags_df)} valid book-tag relationships")
            
            # Load optional files
            if ratings_file.exists():
                print("Loading ratings.csv...")
                self.ratings_df = pd.read_csv(ratings_file)
                print(f"Loaded {len(self.ratings_df)} ratings")
            
            if to_read_file.exists():
                print("Loading to_read.csv...")
                self.to_read_df = pd.read_csv(to_read_file)
                print(f"Loaded {len(self.to_read_df)} to-read entries")
            
            print("CSV data loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            print("Falling back to sample data...")
            return self.load_sample_data()
    
    def load_sample_data(self):
        """Load sample data as fallback"""
        print("Loading sample book data...")
        
        # Sample books data
        books_data = {
            'goodreads_book_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'book_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'title': [
                'The Great Gatsby', 'To Kill a Mockingbird', '1984', 'Pride and Prejudice',
                'The Catcher in the Rye', 'Lord of the Flies', 'The Hobbit', 'Harry Potter and the Sorcerer\'s Stone',
                'The Da Vinci Code', 'Gone Girl', 'The Alchemist', 'Brave New World',
                'The Book Thief', 'Life of Pi', 'The Kite Runner', 'The Girl with the Dragon Tattoo',
                'The Hunger Games', 'The Fault in Our Stars', 'Dune', 'The Lord of the Rings'
            ],
            'authors': [
                'F. Scott Fitzgerald', 'Harper Lee', 'George Orwell', 'Jane Austen',
                'J.D. Salinger', 'William Golding', 'J.R.R. Tolkien', 'J.K. Rowling',
                'Dan Brown', 'Gillian Flynn', 'Paulo Coelho', 'Aldous Huxley',
                'Markus Zusak', 'Yann Martel', 'Khaled Hosseini', 'Stieg Larsson',
                'Suzanne Collins', 'John Green', 'Frank Herbert', 'J.R.R. Tolkien'
            ],
            'average_rating': [4.2, 4.5, 4.3, 4.4, 3.8, 3.9, 4.6, 4.7, 4.0, 4.1, 4.2, 4.0, 4.5, 4.1, 4.3, 4.2, 4.4, 4.3, 4.5, 4.8],
            'ratings_count': [1500000, 2000000, 1800000, 1200000, 900000, 800000, 2200000, 3000000, 1600000, 1400000, 1100000, 950000, 1300000, 800000, 1200000, 1000000, 2500000, 1800000, 600000, 2800000],
            'image_url': [f'/placeholder.svg?height=200&width=150&text=Book+{i}' for i in range(1, 21)]
        }
        
        # Sample tags data
        tags_data = {
            'tag_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'tag_name': [
                'fiction', 'fantasy', 'romance', 'mystery', 'thriller', 'science-fiction',
                'historical-fiction', 'young-adult', 'non-fiction', 'biography', 'memoir',
                'self-help', 'business', 'history', 'philosophy', 'psychology', 'horror',
                'adventure', 'comedy', 'drama'
            ]
        }
        
        # Sample book-tags relationships
        book_tags_data = []
        np.random.seed(42)  # For reproducible results
        
        for book_id in range(1, 21):
            # Each book gets 2-5 random tags
            num_tags = np.random.randint(2, 6)
            selected_tags = np.random.choice(range(1, 21), size=num_tags, replace=False)
            
            for tag_id in selected_tags:
                count = np.random.randint(10, 1000)  # Random tag count
                book_tags_data.append({
                    'goodreads_book_id': book_id,
                    'tag_id': int(tag_id),
                    'count': count
                })
        
        # Create DataFrames
        self.books_df = pd.DataFrame(books_data)
        self.tags_df = pd.DataFrame(tags_data)
        self.book_tags_df = pd.DataFrame(book_tags_data)
        
        print(f"Loaded {len(self.books_df)} books, {len(self.tags_df)} tags, {len(self.book_tags_df)} book-tag relationships")
        return True
        
    def prepare_data(self):
        """Prepare the data for ML algorithms"""
        if self.books_df is None:
            self.load_csv_data()
        
        print("Preparing tag feature matrix...")
        
        # Build pivot table: rows=goodreads_book_id, cols=tag_id, values=count
        self.pivot = self.book_tags_df.pivot_table(
            index="goodreads_book_id",
            columns="tag_id", 
            values="count",
            fill_value=0
        )
        
        print(f"Created pivot matrix with {self.pivot.shape[0]} books and {self.pivot.shape[1]} tags")
        
        # Cache pivot index map for quick lookups
        self.pivot_index_map = {gbid: idx for idx, gbid in enumerate(self.pivot.index)}
        
        # Apply TF-IDF transform
        print("Applying TF-IDF transform...")
        tfidf = TfidfTransformer()
        self.pivot_tfidf = tfidf.fit_transform(self.pivot)
        
        self.data_loaded = True
        print("Data preparation completed successfully.")
    
    def get_genres(self, top_n=50):
        """Get top N popular genres"""
        if not self.data_loaded:
            self.prepare_data()
        
        # Count tag frequency
        tag_counts = self.book_tags_df.groupby("tag_id")["count"].sum()
        top_tags = tag_counts.sort_values(ascending=False).head(top_n).index.tolist()
        
        # Get tag names and filter out invalid ones
        genres = self.tags_df[self.tags_df["tag_id"].isin(top_tags)][["tag_id", "tag_name"]]
        
        # Additional filtering for better genre names
        genres = genres[
            (genres['tag_name'].str.len() > 2) &  # Remove very short names
            (~genres['tag_name'].str.contains(r'^\d+$', regex=True)) &  # Remove pure numbers
            (~genres['tag_name'].str.contains(r'^-+', regex=True))  # Remove names starting with dashes
        ]
        
        return genres.to_dict(orient="records")
    
    def get_book_clusters(self, selected_tag_ids, num_clusters=5):
        """Get book clusters based on selected genres"""
        if not self.data_loaded:
            self.prepare_data()
        
        genre_tag_set = set(selected_tag_ids)
        if len(genre_tag_set) == 0:
            raise ValueError("No tags selected")
        
        # Find books that have any of the selected tags
        books_in_genre = self.book_tags_df[
            self.book_tags_df["tag_id"].isin(genre_tag_set)
        ]["goodreads_book_id"].unique()
        
        if len(books_in_genre) == 0:
            # Try to find similar tags or provide better error message
            available_tags = self.book_tags_df["tag_id"].unique()
            suggested_tags = [tag for tag in available_tags if tag in self.tags_df["tag_id"].values][:5]
            raise ValueError(f"No books found for selected genres. Try using tag IDs from: {suggested_tags}")
        
        print(f"Found {len(books_in_genre)} books for selected tags")
        
        # Filter pivot matrix for these books
        pivot_genre = self.pivot.loc[self.pivot.index.intersection(books_in_genre)]
        
        if len(pivot_genre) == 0:
            raise ValueError("No books found in pivot matrix for selected genres")
        
        # Adjust number of clusters based on available books
        if len(pivot_genre) < num_clusters:
            num_clusters = max(1, len(pivot_genre))
            print(f"Adjusted cluster count to {num_clusters} based on available books")
        
        # TF-IDF transform on subset
        tfidf = TfidfTransformer()
        pivot_genre_tfidf = tfidf.fit_transform(pivot_genre)
        
        # KMeans clustering
        cache_key = "-".join(map(str, sorted(genre_tag_set)))
        if cache_key in self.kmeans_models and len(self.kmeans_models[cache_key].cluster_centers_) == num_clusters:
            kmeans = self.kmeans_models[cache_key]
        else:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            kmeans.fit(pivot_genre_tfidf)
            self.kmeans_models[cache_key] = kmeans
        
        # Assign clusters
        clusters = kmeans.predict(pivot_genre_tfidf)
        pivot_genre = pivot_genre.copy()
        pivot_genre["cluster"] = clusters
        
        # Prepare response
        books_subset = self.books_df.set_index("goodreads_book_id")
        cluster_groups = []
        
        for cluster_num in range(num_clusters):
            cluster_book_gbids = pivot_genre[pivot_genre["cluster"] == cluster_num].index.tolist()
            
            books_info = []
            for gbid in cluster_book_gbids:
                if gbid in books_subset.index:
                    b = books_subset.loc[gbid]
                    
                    # Handle potential missing or invalid data
                    image_url = b.get("image_url", "")
                    if pd.isna(image_url) or image_url == "":
                        image_url = f'/placeholder.svg?height=200&width=150&text={str(b["title"]).replace(" ", "+")}'
                
                    books_info.append({
                        "goodreads_book_id": int(gbid),
                        "book_id": int(b.get("book_id", gbid)),
                        "title": str(b["title"]),
                        "authors": str(b["authors"]),
                        "average_rating": float(b.get("average_rating", 0)),
                        "ratings_count": int(b.get("ratings_count", 0)),
                        "image_url": image_url,
                    })
            
            if books_info:  # Only add non-empty clusters
                cluster_groups.append({
                    "cluster_name": f"Cluster {cluster_num + 1}",
                    "books": books_info
                })
    
        print(f"Generated {len(cluster_groups)} clusters with books")
        return cluster_groups
    
    def get_recommendations(self, favorite_goodreads_book_ids, top_n=10):
        """Get recommendations based on favorite books"""
        if not self.data_loaded:
            self.prepare_data()
        
        user_favorites = set(favorite_goodreads_book_ids)
        
        if not user_favorites:
            raise ValueError("No favorite books provided")
        
        # Check if favorites exist in our data
        missing_favs = [fb for fb in user_favorites if fb not in self.pivot_index_map]
        if missing_favs:
            # Try to find some valid favorites from the available data
            available_favorites = [fb for fb in user_favorites if fb in self.pivot_index_map]
            if not available_favorites:
                raise ValueError(f"None of the favorite book IDs found in data: {favorite_goodreads_book_ids}")
            user_favorites = set(available_favorites)
        
        favorite_indices = [self.pivot_index_map[fb] for fb in user_favorites]
        
        # Compute user profile as average of favorite book vectors
        user_profile = self.pivot_tfidf[favorite_indices].mean(axis=0)
        user_profile_array = np.asarray(user_profile)
        
        # Calculate cosine similarity to all books
        similarities = cosine_similarity(self.pivot_tfidf, user_profile_array).flatten()
        
        # Rank books by similarity, excluding favorites
        ranked_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        books_subset = self.books_df.set_index("goodreads_book_id")
        count = 0
        
        for idx in ranked_indices:
            gbid = self.pivot.index[idx]
            if gbid in user_favorites:
                continue
            if gbid not in books_subset.index:
                continue
            
            b = books_subset.loc[gbid]
            
            # Handle potential missing or invalid data
            image_url = b.get("image_url", "")
            if pd.isna(image_url) or image_url == "":
                image_url = f'/placeholder.svg?height=200&width=150&text={b["title"].replace(" ", "+")}'
            
            recommendations.append({
                "goodreads_book_id": int(gbid),
                "book_id": int(b.get("book_id", gbid)),
                "title": str(b["title"]),
                "authors": str(b["authors"]),
                "average_rating": float(b.get("average_rating", 0)),
                "ratings_count": int(b.get("ratings_count", 0)),
                "image_url": image_url,
            })
            count += 1
            if count >= top_n:
                break
        
        return {"recommendations": recommendations}
    
    def get_book_details(self, goodreads_book_id):
        """Get details for a specific book"""
        if not self.data_loaded:
            self.prepare_data()
        
        if goodreads_book_id not in self.books_df["goodreads_book_id"].values:
            raise ValueError("Book not found")
        
        b = self.books_df[self.books_df["goodreads_book_id"] == goodreads_book_id].iloc[0]
        
        # Handle potential missing or invalid data
        image_url = b.get("image_url", "")
        if pd.isna(image_url) or image_url == "":
            image_url = f'/placeholder.svg?height=200&width=150&text={b["title"].replace(" ", "+")}'
        
        return {
            "goodreads_book_id": int(goodreads_book_id),
            "book_id": int(b.get("book_id", goodreads_book_id)),
            "title": str(b["title"]),
            "authors": str(b["authors"]),
            "average_rating": float(b.get("average_rating", 0)),
            "ratings_count": int(b.get("ratings_count", 0)),
            "image_url": image_url,
        }

# Global instance
recommendation_engine = BookRecommendationEngine()

# Initialize the engine when the module is imported
try:
    recommendation_engine.prepare_data()
    print("Recommendation engine initialized successfully!")
except Exception as e:
    print(f"Error initializing recommendation engine: {e}")
