"""
Script to find the most popular tags that actually have books associated with them
"""

from recommendation_engine import recommendation_engine
import pandas as pd

def find_popular_tags_with_books(top_n=20):
    """Find the most popular tags that have books"""
    
    if not recommendation_engine.data_loaded:
        print("Loading data...")
        recommendation_engine.prepare_data()
    
    print("Finding popular tags with associated books...")
    
    # Get tag popularity with book counts
    tag_stats = recommendation_engine.book_tags_df.groupby('tag_id').agg({
        'goodreads_book_id': 'nunique',  # Number of unique books
        'count': 'sum'  # Total tag occurrences
    }).sort_values('goodreads_book_id', ascending=False)
    
    # Merge with tag names
    tag_stats = tag_stats.merge(
        recommendation_engine.tags_df[['tag_id', 'tag_name']], 
        left_index=True, 
        right_on='tag_id', 
        how='left'
    )
    
    # Filter out problematic tag names
    tag_stats = tag_stats[
        (tag_stats['tag_name'].notna()) &
        (tag_stats['tag_name'].str.len() > 2) &
        (~tag_stats['tag_name'].str.contains(r'^\d+$', regex=True, na=False)) &
        (~tag_stats['tag_name'].str.contains(r'^-+', regex=True, na=False)) &
        (~tag_stats['tag_name'].str.contains('books-i-own', na=False)) &
        (~tag_stats['tag_name'].str.contains('currently-reading', na=False)) &
        (~tag_stats['tag_name'].str.contains('to-read', na=False))
    ]
    
    top_tags = tag_stats.head(top_n)
    
    print(f"\n📊 Top {len(top_tags)} Popular Tags with Books:")
    print("=" * 60)
    
    for _, row in top_tags.iterrows():
        print(f"Tag ID: {row['tag_id']:>6} | Books: {row['goodreads_book_id']:>5} | Name: '{row['tag_name']}'")
    
    return top_tags

def test_with_popular_tags():
    """Test the recommendation system with popular tags"""
    print("\n🧪 Testing with Popular Tags")
    print("=" * 40)
    
    popular_tags = find_popular_tags_with_books(10)
    
    if len(popular_tags) < 3:
        print("❌ Not enough popular tags found")
        return False
    
    # Use top 3 tags for testing
    test_tag_ids = popular_tags.head(3)['tag_id'].tolist()
    test_tag_names = popular_tags.head(3)['tag_name'].tolist()
    
    print(f"\n🎯 Testing with tags: {test_tag_names}")
    print(f"   Tag IDs: {test_tag_ids}")
    
    try:
        # Test clustering
        clusters = recommendation_engine.get_book_clusters(test_tag_ids, 3)
        print(f"✅ Successfully created {len(clusters)} clusters")
        
        # Get sample books for recommendations
        sample_books = []
        for cluster in clusters:
            if cluster['books']:
                sample_books.extend([book['goodreads_book_id'] for book in cluster['books'][:2]])
            if len(sample_books) >= 4:
                break
        
        if sample_books:
            # Test recommendations
            recommendations = recommendation_engine.get_recommendations(sample_books[:3], 5)
            print(f"✅ Successfully generated {len(recommendations['recommendations'])} recommendations")
            
            print("\n📚 Sample Recommendations:")
            for i, rec in enumerate(recommendations['recommendations'][:3], 1):
                print(f"   {i}. '{rec['title']}' by {rec['authors']}")
            
            return True
        else:
            print("❌ No books found in clusters")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function"""
    print("🔍 Finding Popular Tags for LitWise")
    print("=" * 40)
    
    try:
        # Find popular tags
        popular_tags = find_popular_tags_with_books(20)
        
        # Test with popular tags
        success = test_with_popular_tags()
        
        if success:
            print("\n🎉 Popular tags test completed successfully!")
            print("\n💡 Recommended tags for frontend:")
            top_5 = popular_tags.head(5)
            for _, row in top_5.iterrows():
                print(f"   {row['tag_id']}: '{row['tag_name']}' ({row['goodreads_book_id']} books)")
        else:
            print("\n⚠️  Some issues found with popular tags")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
