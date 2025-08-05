import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Book Recommendation Engine Backend")

# Data holders to be populated on startup
books_df = None
tags_df = None
book_tags_df = None
ratings_df = None
pivot = None                 # Tag feature matrix (books x tags)
pivot_tfidf = None           # TF-IDF transformed matrix
pivot_index_map = None       # Maps goodreads_book_id to row index in pivot
kmeans_models = dict()       # Cache KMeans models for genres

# Pydantic models for requests/responses
class GenreResponse(BaseModel):
    tag_id: int
    tag_name: str

class BookInfo(BaseModel):
    goodreads_book_id: int
    book_id: int
    title: str
    authors: str
    average_rating: float
    ratings_count: int
    image_url: str

class ClusteredBooksResponse(BaseModel):
    cluster_name: str
    books: List[BookInfo]

class FavoriteBooksRequest(BaseModel):
    favorite_goodreads_book_ids: List[int]

class RecommendationsResponse(BaseModel):
    recommendations: List[BookInfo]

@app.on_event("startup")
def load_and_prepare_data():
    global books_df, tags_df, book_tags_df, ratings_df, pivot, pivot_tfidf, pivot_index_map

    print("Loading datasets...")
    books_df = pd.read_csv("books.csv")
    tags_df = pd.read_csv("tags.csv")
    book_tags_df = pd.read_csv("book_tags.csv")
    ratings_df = pd.read_csv("ratings.csv")

    # Build pivot table: rows=goodreads_book_id, cols=tag_id, values=count
    print("Preparing tag feature matrix...")
    pivot = book_tags_df.pivot_table(
        index="goodreads_book_id",
        columns="tag_id",
        values="count",
        fill_value=0
    )

    # Cache pivot index map for quick index lookups later
    pivot_index_map = {gbid: idx for idx, gbid in enumerate(pivot.index)}

    # Apply TF-IDF transform globally for reuse
    print("Applying TF-IDF transform...")
    tfidf = TfidfTransformer()
    pivot_tfidf = tfidf.fit_transform(pivot)

    print("Data loaded and preprocessed successfully.")


@app.get("/genres", response_model=List[GenreResponse])
def get_genres(top_n: Optional[int] = 50):
    """
    Returns the top N popular genres (tags) sorted by frequency.
    """
    # Count tag frequency in book_tags_df
    tag_counts = book_tags_df.groupby("tag_id")["count"].sum()
    top_tags = tag_counts.sort_values(ascending=False).head(top_n).index.tolist()

    genres = tags_df[tags_df["tag_id"].isin(top_tags)][["tag_id", "tag_name"]]

    # Optionally filter non-genre tags (like 'to-read') if needed here
    # You can implement filtering logic based on tag_name or tag_id

    return genres.to_dict(orient="records")


@app.get("/clusters", response_model=List[ClusteredBooksResponse])
def get_book_clusters(selected_tag_ids: List[int], num_clusters: Optional[int] = 5):
    """
    Given user-selected genre tag IDs, returns books clustered by tags within that genre.
    """
    genre_tag_set = set(selected_tag_ids)
    if len(genre_tag_set) == 0:
        raise HTTPException(status_code=400, detail="No tags selected")

    # Find books that have any of the selected tags
    books_in_genre = book_tags_df[book_tags_df["tag_id"].isin(genre_tag_set)]["goodreads_book_id"].unique()
    if len(books_in_genre) == 0:
        raise HTTPException(status_code=404, detail="No books found for selected genres")

    # Filter pivot matrix for these books
    pivot_genre = pivot.loc[pivot.index.intersection(books_in_genre)]

    # TF-IDF transform on subset
    tfidf = TfidfTransformer()
    pivot_genre_tfidf = tfidf.fit_transform(pivot_genre)

    # KMeans clustering (cache by tag combination string for efficiency)
    cache_key = "-".join(map(str, sorted(genre_tag_set)))
    if cache_key in kmeans_models:
        kmeans = kmeans_models[cache_key]
    else:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(pivot_genre_tfidf)
        kmeans_models[cache_key] = kmeans

    # Assign clusters
    clusters = kmeans.predict(pivot_genre_tfidf)
    pivot_genre = pivot_genre.copy()
    pivot_genre["cluster"] = clusters

    # Prepare response: group books by cluster
    books_subset = books_df.set_index("goodreads_book_id")
    cluster_groups = []
    for cluster_num in range(num_clusters):
        cluster_book_gbids = pivot_genre[pivot_genre["cluster"] == cluster_num].index.tolist()

        # Fetch book info for each book
        books_info = []
        for gbid in cluster_book_gbids:
            if gbid in books_subset.index:
                b = books_subset.loc[gbid]
                books_info.append(BookInfo(
                    goodreads_book_id=gbid,
                    book_id=int(b["book_id"]),
                    title=b["title"],
                    authors=b["authors"],
                    average_rating=float(b["average_rating"]),
                    ratings_count=int(b["ratings_count"]),
                    image_url=b["image_url"] if "image_url" in b and pd.notna(b["image_url"]) else "",
                ))
        cluster_groups.append(ClusteredBooksResponse(
            cluster_name=f"Cluster {cluster_num + 1}",
            books=books_info
        ))

    return cluster_groups


@app.post("/recommendations", response_model=RecommendationsResponse)
def get_recommendations(request: FavoriteBooksRequest, top_n: Optional[int] = 10):
    """
    Given a list of user's favorite goodreads_book_ids, returns recommendations using content-based filtering.
    """
    user_favorites = set(request.favorite_goodreads_book_ids)

    # Validate input
    if not user_favorites:
        raise HTTPException(status_code=400, detail="No favorite books provided")

    missing_favs = [fb for fb in user_favorites if fb not in pivot_index_map]
    if missing_favs:
        raise HTTPException(status_code=400, detail=f"These favorite book IDs not found in data: {missing_favs}")

    favorite_indices = [pivot_index_map[fb] for fb in user_favorites]

    # Compute user profile vector as average of favorite book tag TF-IDF vectors
    user_profile = pivot_tfidf[favorite_indices].mean(axis=0)
    user_profile_array = np.asarray(user_profile)

    # Calculate cosine similarity to all books
    similarities = cosine_similarity(pivot_tfidf, user_profile_array).flatten()

    # Rank books by similarity excluding favorites
    ranked_indices = np.argsort(similarities)[::-1]

    recommendations = []
    books_subset = books_df.set_index("goodreads_book_id")
    count = 0
    for idx in ranked_indices:
        gbid = pivot.index[idx]
        if gbid in user_favorites:
            continue
        if gbid not in books_subset.index:
            continue
        b = books_subset.loc[gbid]
        recommendations.append(BookInfo(
            goodreads_book_id=gbid,
            book_id=int(b["book_id"]),
            title=b["title"],
            authors=b["authors"],
            average_rating=float(b["average_rating"]),
            ratings_count=int(b["ratings_count"]),
            image_url=b["image_url"] if "image_url" in b and pd.notna(b["image_url"]) else "",
        ))
        count += 1
        if count >= top_n:
            break

    return RecommendationsResponse(recommendations=recommendations)


@app.get("/book/{goodreads_book_id}", response_model=BookInfo)
def get_book_details(goodreads_book_id: int):
    """
    Returns detailed info of a single book by Goodreads Book ID.
    """
    if goodreads_book_id not in books_df["goodreads_book_id"].values:
        raise HTTPException(status_code=404, detail="Book not found")

    b = books_df[books_df["goodreads_book_id"] == goodreads_book_id].iloc[0]
    return BookInfo(
        goodreads_book_id=goodreads_book_id,
        book_id=int(b["book_id"]),
        title=b["title"],
        authors=b["authors"],
        average_rating=float(b["average_rating"]),
        ratings_count=int(b["ratings_count"]),
        image_url=b["image_url"] if "image_url" in b and pd.notna(b["image_url"]) else "",
    )
