# from engine.bsbi import BSBIIndex
# from engine.compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
import os
from bsbi import BSBIIndex
from compression import VBEPostings


# BSBI_instance = BSBIIndex(data_dir='collections',
#                           postings_encoding=VBEPostings,
#                           output_dir='index')
output_dir = os.path.join(os.getcwd(), "index")
data_dir = os.path.join(os.getcwd(),  "collection")
BSBI_instance = BSBIIndex(data_dir=data_dir,
                        postings_encoding=VBEPostings,
                        output_dir=output_dir)
BSBI_instance.load()

queries = ["negara"]
for query in queries:
    print("Query  : ", query)
    print("Results:")
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k=10):
        print(f"{doc:30} {score:>.3f}")
    print()
