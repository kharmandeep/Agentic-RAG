import weaviate
from weaviate.classes.config import Property,DataType,Configure


collection_name = "Documents"

with weaviate.connect_to_local() as client:
    exists = client.collections.exists(collection_name)
    if exists:
        client.collections.delete(collection_name)
        print(f"Deleted existing collection '{collection_name}'")

    res = client.collections.create(
        collection_name,
        vector_config=[Configure.Vectors.text2vec_transformers(
            name="embedding",
            source_properties=["content"]
        )],
        # Enable BM25 for hybrid search
        inverted_index_config=Configure.inverted_index(
            bm25_b=0.75,
            bm25_k1=1.2,
            cleanup_interval_seconds=60,
        ),
        properties=[
            # Core essentials
            Property(name="content", data_type=DataType.TEXT, index_searchable=True),
            Property(name="source", data_type=DataType.TEXT, index_filterable=True),
            Property(name="domain", data_type=DataType.TEXT, index_filterable=True),
            Property(name="doc_type", data_type=DataType.TEXT, index_filterable=True),
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
            
            # Optional metadata
            Property(name="section_title", data_type=DataType.TEXT),
        ]
    )
    print(f"Collection {collection_name} created, Metadata: {res}")