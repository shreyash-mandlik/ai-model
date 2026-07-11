from src .docloader import Mainloader
from src.chunking import Chunker
from src.vectorization import Vectorizer
from src.retriever import mainretriever

mn = Mainloader()
ch = Chunker()
vc = Vectorizer()
mnr = mainretriever()

documents = mn.allfileloader()
print("test1")
chunked_doc = ch.recursiveoverlap(documents)
print("test2")
vector_db = vc.hfembedder(chunked_doc)
print("test3")
fncontext = mnr.Topkretriever("What is Tesla", vector_db, 3)

print(fncontext)
