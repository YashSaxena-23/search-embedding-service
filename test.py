import base64
import struct

from EmbeddingTest import embedding_service

# Test your specific case
embedding1 = embedding_service.getEmbedding("Richa Madan IOS Testing-121")
embedding2 = embedding_service.getEmbedding("rich madan ios")
print(f" embedding1: {embedding2}")

binary_data = struct.pack('f' * len(embedding2), *embedding2)
encoded_vector = base64.b64encode(binary_data).decode('utf-8')
print('Encoded vector for Redis:')
print(encoded_vector)


similarity = embedding_service.calculate_similarity(embedding1, embedding2)
print(f"Similarity: {similarity:.4f}")

if similarity > 0.7:
    print("✅ Vector search should work!")
else:
    print("❌ Use text search instead")