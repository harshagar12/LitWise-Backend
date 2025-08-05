"""
Test script to verify the recommendation engine works correctly
"""

from recommendation_engine import recommendation_engine
import json

def test_genres():
    """Test genre fetching"""
    print("Testing genre fetching...")
    try:
        genres = recommendation_engine.get_genres(20)
        print(f"✓ Successfully fetched {len(genres)} genres")
        print("Sample genres:", genres[:3])
        return True, genres
    except Exception as e:
        print(f"✗ Genre test failed: {e}")
        return False, []

def test_clusters(genres):
    """Test book clustering"""
    print("\nTesting book clustering...")
    try:
        if not genres:
            print("✗ No genres available for clustering test")
            return False, []
        
        # Find genres that actually have books associated with them
        valid_genres = []
        for genre in genres:
            tag_id = genre['tag_id']
            # Check if this tag has books
            books_with_tag = recommendation_engine.book_tags_df[
                recommendation_engine.book_tags_df['tag_id'] == tag_id
            ]['goodreads_book_id'].nunique()
            
            if books_with_tag > 0:
                valid_genres.append(genre)
                print(f"   Tag '{genre['tag_name']}' has {books_with_tag} books")
            
            if len(valid_genres) >= 3:  # We only need 3 for testing
                break
        
        if len(valid_genres) < 1:
            print("✗ No valid genres found with associated books")
            return False, []
        
        # Use valid genres for clustering
        sample_tag_ids = [g['tag_id'] for g in valid_genres[:3]]
        print(f"   Using tag IDs: {sample_tag_ids}")
        
        clusters = recommendation_engine.get_book_clusters(sample_tag_ids, 3)
        print(f"✓ Successfully generated {len(clusters)} clusters")
        
        total_books = sum(len(cluster['books']) for cluster in clusters)
        print(f"   Total books in clusters: {total_books}")
        
        for i, cluster in enumerate(clusters):
            print(f"   Cluster {i+1}: {len(cluster['books'])} books")
            if cluster['books']:
                sample_book = cluster['books'][0]
                print(f"      Sample: '{sample_book['title']}' by {sample_book['authors']}")
        
        return True, clusters
    except Exception as e:
        print(f"✗ Clustering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def test_recommendations(clusters):
    """Test recommendation generation"""
    print("\nTesting recommendations...")
    try:
        # Get sample book IDs from clusters
        sample_book_ids = []
        for cluster in clusters:
            if cluster['books']:
                sample_book_ids.extend([book['goodreads_book_id'] for book in cluster['books'][:2]])
            if len(sample_book_ids) >= 3:  # We only need a few for testing
                break
        
        if not sample_book_ids:
            # Fallback: get some book IDs directly from the dataset
            sample_book_ids = recommendation_engine.books_df['goodreads_book_id'].head(3).tolist()
            print(f"   Using fallback book IDs: {sample_book_ids}")
        else:
            print(f"   Using book IDs from clusters: {sample_book_ids[:3]}")
        
        recommendations = recommendation_engine.get_recommendations(sample_book_ids[:3], 5)
        print(f"✓ Successfully generated {len(recommendations['recommendations'])} recommendations")
        
        for i, rec in enumerate(recommendations['recommendations'][:3], 1):
            print(f"   {i}. '{rec['title']}' by {rec['authors']} (Rating: {rec['average_rating']:.2f})")
        
        return True
    except Exception as e:
        print(f"✗ Recommendations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_quality():
    """Test data quality and provide insights"""
    print("\nTesting data quality...")
    try:
        # Check book-tag relationships
        book_tags_df = recommendation_engine.book_tags_df
        books_df = recommendation_engine.books_df
        tags_df = recommendation_engine.tags_df
        
        # Find most popular tags with books
        popular_tags = book_tags_df.groupby('tag_id').agg({
            'goodreads_book_id': 'nunique',
            'count': 'sum'
        }).sort_values('goodreads_book_id', ascending=False).head(10)
        
        print("✓ Most popular tags with books:")
        for tag_id, row in popular_tags.iterrows():
            tag_name = tags_df[tags_df['tag_id'] == tag_id]['tag_name'].iloc[0] if len(tags_df[tags_df['tag_id'] == tag_id]) > 0 else f"Tag {tag_id}"
            print(f"   '{tag_name}': {row['goodreads_book_id']} books, {row['count']} total occurrences")
        
        # Check books with most tags
        books_with_tags = book_tags_df.groupby('goodreads_book_id').size().sort_values(ascending=False).head(5)
        print("\n✓ Books with most tags:")
        for book_id, tag_count in books_with_tags.items():
            book_title = books_df[books_df['goodreads_book_id'] == book_id]['title'].iloc[0] if len(books_df[books_df['goodreads_book_id'] == book_id]) > 0 else f"Book {book_id}"
            print(f"   '{book_title}': {tag_count} tags")
        
        return True
    except Exception as e:
        print(f"✗ Data quality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Book Recommendation Engine")
    print("=" * 40)
    
    # Test data quality first
    data_quality_passed = test_data_quality()
    
    # Run main tests
    genres_passed, genres = test_genres()
    clusters_passed, clusters = test_clusters(genres)
    recommendations_passed = test_recommendations(clusters)
    
    tests = [data_quality_passed, genres_passed, clusters_passed, recommendations_passed]
    passed = sum(tests)
    
    print(f"\n{'=' * 40}")
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("✅ All tests passed! Recommendation engine is working correctly.")
    elif passed >= 3:
        print("⚠️  Most tests passed. The recommendation engine is functional with minor issues.")
    else:
        print("❌ Several tests failed. Check the error messages above.")
    
    print("\n📊 Dataset Summary:")
    print(f"   Books: {len(recommendation_engine.books_df):,}")
    print(f"   Tags: {len(recommendation_engine.tags_df):,}")
    print(f"   Book-Tag relationships: {len(recommendation_engine.book_tags_df):,}")
    if recommendation_engine.ratings_df is not None:
        print(f"   Ratings: {len(recommendation_engine.ratings_df):,}")
    if recommendation_engine.to_read_df is not None:
        print(f"   To-read entries: {len(recommendation_engine.to_read_df):,}")

if __name__ == "__main__":
    main()
