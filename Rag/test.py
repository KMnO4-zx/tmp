from Embeddings import BaseEmbeddings, OpenAIEmbedding, JinaEmbedding

embedding = JinaEmbedding(path="G:\日常文件\model\jinaai\jina-embeddings-v2-base-zh")
response1 = embedding.get_embedding('我喜欢你')
response2 = embedding.get_embedding('我钟意你')
print('Jina: ',embedding.cosine_similarity(response1, response2))

embedding_openai = OpenAIEmbedding()
response3 = embedding_openai.get_embedding('我喜欢你')
response4 = embedding_openai.get_embedding('我钟意你')
print('OpenAI: ',embedding_openai.cosine_similarity(response3, response4))