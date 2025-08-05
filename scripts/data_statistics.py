"""
Script to analyze and display statistics about the loaded dataset
"""

from recommendation_engine import recommendation_engine
import pandas as pd

def analyze_dataset():
    """Analyze the loaded dataset and display statistics"""
    
    if not recommendation_engine.data_loaded:
        print("Loading data...")
        recommendation_engine.prepare_data()
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    # Books statistics
    if recommendation_engine.books_df is not None:
        books_df = recommendation_engine.books_df
        print(f"\n📚 BOOKS:")
        print(f"   Total books: {len(books_df):,}")
        print(f"   Average rating range: {books_df['average_rating'].min():.2f} - {books_df['average_rating'].max():.2f}")
        print(f"   Total ratings: {books_df['ratings_count'].sum():,}")
        print(f"   Most rated book: {books_df.loc[books_df['ratings_count'].idxmax(), 'title']}")
        print(f"   Highest rated book: {books_df.loc[books_df['average_rating'].idxmax(), 'title']}")
    
    # Tags statistics
    if recommendation_engine.tags_df is not None:
        tags_df = recommendation_engine.tags_df
        print(f"\n🏷️  TAGS:")
        print(f"   Total tags: {len(tags_df):,}")
        print(f"   Sample tags: {', '.join(tags_df['tag_name'].head(10).tolist())}")
    
    # Book-Tags relationships
    if recommendation_engine.book_tags_df is not None:
        book_tags_df = recommendation_engine.book_tags_df
        print(f"\n🔗 BOOK-TAG RELATIONSHIPS:")
        print(f"   Total relationships: {len(book_tags_df):,}")
        print(f"   Average tags per book: {len(book_tags_df) / len(books_df):.1f}")
        
        # Most popular tags
        popular_tags = book_tags_df.groupby('tag_id')['count'].sum().sort_values(ascending=False).head(10)
        tag_names = recommendation_engine.tags_df.set_index('tag_id')['tag_name']
        
        print(f"\n   📈 Most popular tags:")
        for tag_id, count in popular_tags.items():
            tag_name = tag_names.get(tag_id, f"Tag {tag_id}")
            print(f"      {tag_name}: {count:,} occurrences")
    
    # Ratings statistics (if available)
    if recommendation_engine.ratings_df is not None:
        ratings_df = recommendation_engine.ratings_df
        print(f"\n⭐ RATINGS:")
        print(f"   Total ratings: {len(ratings_df):,}")
        print(f"   Unique users: {ratings_df['user_id'].nunique():,}")
        print(f"   Average rating: {ratings_df['rating'].mean():.2f}")
        print(f"   Rating distribution:")
        rating_dist = ratings_df['rating'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            print(f"      {rating} stars: {count:,} ({count/len(ratings_df)*100:.1f}%)")
    
    # ML Model statistics
    if recommendation_engine.pivot is not None:
        pivot = recommendation_engine.pivot
        print(f"\n🤖 ML MODEL:")
        print(f"   Feature matrix size: {pivot.shape[0]:,} books × {pivot.shape[1]:,} tags")
        print(f"   Sparsity: {(pivot == 0).sum().sum() / (pivot.shape[0] * pivot.shape[1]) * 100:.1f}%")
        print(f"   Cached clustering models: {len(recommendation_engine.kmeans_models)}")

def test_recommendations():
    """Test the recommendation system with sample data"""
    print("\n" + "="*60)
    print("TESTING RECOMMENDATION SYSTEM")
    print("="*60)
    
    try:
        # Test genres
        print("\n🧪 Testing genre fetching...")
        genres = recommendation_engine.get_genres(10)
        print(f"   ✅ Successfully fetched {len(genres)} genres")
        
        # Test clustering
        print("\n🧪 Testing book clustering...")
        if genres:
            sample_tag_ids = [g['tag_id'] for g in genres[:3]]
            clusters = recommendation_engine.get_book_clusters(sample_tag_ids, 3)
            print(f"   ✅ Successfully generated {len(clusters)} clusters")
            
            # Test recommendations
            print("\n🧪 Testing recommendations...")
            if clusters and clusters[0]['books']:
                sample_book_ids = [book['goodreads_book_id'] for book in clusters[0]['books'][:2]]
                recommendations = recommendation_engine.get_recommendations(sample_book_ids, 5)
                print(f"   ✅ Successfully generated {len(recommendations['recommendations'])} recommendations")
            else:
                print("   ⚠️  No books available for recommendation testing")
        else:
            print("   ⚠️  No genres available for testing")
            
    except Exception as e:
        print(f"   ❌ Error during testing: {e}")

def main():
    """Main function to analyze dataset and test system"""
    print("📊 Analyzing LitWise dataset...")
    
    try:
        analyze_dataset()
        test_recommendations()
        
        print("\n" + "="*60)
        print("✅ Dataset analysis completed successfully!")
        print("The recommendation engine is ready to use.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("Please check your data files and try again.")

if __name__ == "__main__":
    main()
