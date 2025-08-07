from EmbeddingTest import embedding_service

# Test your specific case
embedding1 = embedding_service.getEmbedding("Richa Madan IOS Testing-121")
embedding2 = embedding_service.getEmbedding("rich madan ios")

similarity = embedding_service.calculate_similarity(embedding1, embedding2)
print(f"Similarity: {similarity:.4f}")

if similarity > 0.7:
    print("✅ Vector search should work!")
else:
    print("❌ Use text search instead")