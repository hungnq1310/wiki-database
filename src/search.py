import pymilvus
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections, list_collections
from pprint import pprint
import time

connections.connect(host="localhost", port="19530")
print(f"list_collections: {list_collections()}")

wiki_tb = Collection("wiki_tb")
wiki_tb.load()

model = SentenceTransformer('BAAI/bge-base-en-v1.5', device="cuda:0")

input_user = input("USER: ") 

startTime_1 = int(round(time.time() * 1000))

embeddings = model.encode(
    input_user,
    show_progress_bar=True,
    device="cuda:0"
)
print(embeddings.shape)

search_params = {
    "metric_type": "COSINE",
    "offset": 10,
    "params": {"nprobe": 12},
}

startTime_2 = int(round(time.time() * 1000))

result = wiki_tb.search(
    [embeddings], 
    anns_field="embedding",
    param=search_params, 
    limit=3, 
    output_fields=["text"]
)

endTime_2 = int(round(time.time() * 1000))
print(f"Time for search: {endTime_2 - startTime_2} ms")

for e in result[0]:
    # e: {"_hits": <obj_hits>}
    print("text: ", e.entity.text)
print("id: ",result[0].ids)
print("distance: ", result[0].distances)

endTime_1 = int(round(time.time() * 1000))
print(f"Time for whole pipeline: {endTime_1 - startTime_1} ms")
