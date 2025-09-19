# test_vectorization.py
from chroma_embeddings import VectorStoreManager
import asyncio
import time

def test_advanced_vectorization():
    print("🧪 Testing advanced vectorization...")
    
    # Initialize vector store
    vector_store = VectorStoreManager()
    
    # Rebuild the index
    print("🔄 Rebuilding index...")
    start_time = time.time()
    vector_store.index_profiles()
    end_time = time.time()
    print(f"✅ Indexing completed in {end_time - start_time:.2f} seconds")
    
    # Test advanced queries
    test_queries = [
        "temperature measurements in the Pacific Ocean",
        "salinity data from Argo floats in Atlantic",
        "recent profiles with high quality data",
        "deep water measurements with low temperature",
        "coastal monitoring data near continents"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = vector_store.search_profiles(query, n_results=3)
        
        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result['score']:.4f}")
            print(f"     Platform: {result['metadata']['platform_number']}, Cycle: {result['metadata']['cycle_number']}")
            print(f"     Location: {result['metadata']['latitude']:.2f}, {result['metadata']['longitude']:.2f}")
            print(f"     Region: {result['metadata'].get('ocean_region', 'Unknown')}")
            print(f"     Summary: {result['document'][:80]}...")
    
    # Test similar profiles
    print("\n👥 Testing similar profiles...")
    sample_profile_id = results[0]['id'] if results else None
    if sample_profile_id:
        similar = vector_store.get_similar_profiles(sample_profile_id, 3)
        print(f"Profiles similar to {sample_profile_id}:")
        for i, sim in enumerate(similar):
            print(f"  {i+1}. {sim['id']} (score: {sim['score']:.4f})")
    
    # Get stats
    stats = vector_store.get_collection_stats()
    print(f"\n📊 Vector store stats: {stats}")

if __name__ == "__main__":
    test_advanced_vectorization()