# Embeddings, Vector Databases & Search
## Passing context to LMs helps factual recall
Let's first understand why do we even pass context to these language models? As I mentioned, fine-tuning is another way for models to learn knowledge. It is also usually better suited for when you have specialized tasks. So imagine that now you are studying for an exam two weeks away, you might forget about the nitty-gritty details but you probably would have internalized the details a bit better. On the other hand, passing contacts to the model is like when you take an example open notes. 

So it'll help you to be more precise because you have the facts right at hand to reference. The downside, though, is that there is a context length limitation for OpenAI's GPT-3.5 model. **It allows up to 4000 tokens, which is approximately 5 pages of text**.
It is really not very long, considering most documents out there or even just take an example of employee handbook can be easily over five pages. So a really common workaround is to pass and document summaries but maybe also splitting documents into chunks; and that is another strategy that we'll talk about later. 

The latest model by [Anthropic called Claude](https://techcrunch.com/2023/05/11/anthropics-latest-model-can-take-the-great-gatsby-as-input) can accommodate up to a hundred thousand tokens per context. It is in preview as of mid-may 2023 and with the ability to pass in longer context, it probably goes without saying that each API call comes with a higher cost and the model will also need longer time to process longer context as well. So. many researchers believe that by increasing the context length alone doesn't actually help the model to retain information between sessions and unlike human brains, it treats every single piece of information in the context as equally important, so **we will likely need new model architectures to actually solve context problems**.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/3e81fab1-ad7a-4e4b-ac87-42a6c589f0a9)

You saw this slide before in the primer. The whole reason we are talking about vector search is because we convert our context or our knowledge into embedding vectors first before we can do any similarity search. We're also seeing increasing popularity of vector databases. If 2021 was the year of graph databases, 2023 is probably the year of vector databases. But why is that?

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/b37da936-c3b8-41d0-9dd9-d5e4f914093a)

Because vector databases is useful not only for text use cases it's useful for other types of unstructured data as well including for your imaging data and also for your audio data as well. We convert these images or audio files into embedding vectors. We can persist them in a vector database and retrieve them for a variety of tasks. So in this slide, you can see that the tasks really range a really wide realm going from object detection to product search to translation question answering to music transcription and to even identifying machinery malfunction.

## Usecases of Vector Databases
Let's go through a few more example use cases on vector databases. 
- Similarity search: text, images, audio
  - De-duplication
  - Semantic match, rather than keyword match!
    - [Example on enhancing product search](https://www.databricks.com/blog/enhancing-product-search-large-language-models-llms.html)
  - Very useful for knowledge-based Q/A
- Recommendation engines
  - [Example blog post](https://engineering.atspotify.com/2022/03/introducing-natural-language-search-for-podcast-episodes/): Spotify uses vector search to recommend podcast episodes
- Finding security threats
  - Vectorizing virus binaries and finding anomalies

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/24779dc5-b949-48c2-9a28-7865d0ead153)

So with vector search we can calculate the similarity between the vectors. This is incredibly helpful to build knowledge-based Q and A systems and it also gives us the ability to find duplicate items as well.

## Search and Retrieval - Augumented Generation
Now that we have surveyed the landscape of vector search and vector databases can be useful, let's go over the workflow of how you actually implement a Q/A system.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/cbcd243b-a51e-4200-a7c7-9b72537cdf8a)

A knowledge-based question-answering system typically consists of two components: search and also retrieval. But to begin with, a question-answering system also assumes that you have some knowledge base that you can work from, which is the red box in this diagram over here.

So we first have to convert our knowledge base of documents into embedding vectors and then we store these embedding vectors into a vector index either through a vector library or vector database.
A vector index is simply a data structure that facilitates the vector search process and we'll talk a little bit more about vector index later on too. And we also discussed the difference between a vector library and also a vector database. But for now, you can think of all these vector storage solutions as a vector store. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/3ed48631-8ce7-48f3-86c3-04e9d555d6e4)

So now that you have all these documents stored away as embedding vectors in the database or a library, the next thing that you can do is, finally, we can ask users to submit queries against them. Any user queries that we type in natural language will also have to be converted into embedding vectors by a language model.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/f4d6252b-9acb-4887-b4ee-05cda0900f05)

After that, we can now search through the vector index that contains our document embedding vectors and then return the text that is relevant to our user queries. So this step where we determine which documents are relevant or similar to user queries is the search component in this workflow.
Lastly, after we have retrieved all the relevant documents we can call these documents as context or knowledge. We will then pass in this context in a prompt to the language model. So, it means that our language model now will receive a query that is augmented by the context and eventually generates a text output that incorporates the context. 

This entire workflow is called a search and retrieval-augmented generation workflow because the output generated by the language model is augmented by the context that we retrieve during the search process.

# How does Vector Search work?
In vector search, there are two main strategies: exact search and approximate search. 

As the name implies, exact search means that you are using a brute for method to find your nearest neighbors; there's no room or very little room for error. And this is exactly what the conventional KNN does generally. 

With ANN (approximate nearest neighbor) search, you are finding less accurate nearest neighbors but you are gaining in speed. So below is a list of common indexing algorithms. We can call them indexing algorithms because the output of these algorithms is a data structure called a **vector index**. So as we mentioned in the earlier segment, a vector index helps you to hold all the necessary information to conduct an efficient vector search.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/d3256de1-7bb7-4103-b0d1-569d86f83f43)

- K-nearest neighbors (KNN)
- Approximate nearest neighbors (ANN)
  - Trade accuracy for speed gains
  - Examples of indexing algorithms:
    - Tree-based: [ANNOY](https://github.com/spotify/annoy) by Spotify
    - Proximity graphs: [HNSW](https://arxiv.org/abs/1603.09320)
    - Clustering: [FAISS](https://arxiv.org/abs/1603.09320) by Facebook
    - Hashing: [LSH](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)
    - Vector compression: [SCaNN](https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html) by Google

Among all of these algorithms, we see that they can either span from using tree-based methods, clustering to hashing. And we'll cover two of them: **FAISS** and **HNSW**, which are two of the most popular algorithms implemented by vector stores.

## How do we actually determine if two vectors are similar?
The answer is using distance or similarity metrics. And this is probably not a very foreign concept to a lot of you. For distance metrics, we commonly see L1 Manhattan distance or L2 Euclidean distance. Euclidean distance is often the more popular choice.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/14283b69-0620-4bee-b55b-ea051356782c)

So as you can tell, when the distance metric gets higher, then the less similar the vectors will be. On the flip side, we can also measure similarity between vectors by using cosine similarity measure. When you have a higher similarity metric, it means that you have more similar vectors.

It's also worth calling out that when you use either of this L2 distance or cosine similarity on normalized embeddings, then they produce functionally equivalent ranking distances for your vectors. If you are interested in that, feel free to search for mathematical proof online.

## Compressing vectors with Product Quantization
Dense embedding vectors usually take up a lot of space. A common method to reduce that memory usage is to compress the vectors using product quantization, abbreviated as PQ.
This fancy method, called PQ, it really just essentially reduces the number of bytes.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/f5d53ae5-027e-4dab-812d-1386ecf170f3)

And quantization refers to how we represent the vectors using a smaller set of vectors.
So very naively speaking, quantization means that you can either round down or round up a number. But in the context of nearest neighbor search, we start with the original big vector and then we split the big vector into segments of subvectors. And each subvector is then quantized independently and then mapped to the nearest centroid. 

So say that the first subvector is closest to the first centroid, so centroid one. Then, we will replace the vector value with a value of 1. So now you can start to see how we can actually reduce the number
of bytes: instead of storing many floats, we are storing a single integer value.

We'll now move on to talking a little bit more about the vector indexing algorithms.

## FAISS: Facebook AI Similarity Search
It's a clustering algorithm that computes L2 Euclidean distance between the query vectors and all the other points. And as you can imagine, the computation time will only increase as you have more and more vectors. So to optimize the search process, FAISS makes use of something called **Voronoi cells**. What this does is that, instead of computing the distance between every single vector that you have in the storage and the query vector, FAISS actually computes the distance between the query vector and the centroid first. Once it identifies the closest centroid to the query vector, then it will find
all the other vectors similar to that query vector that exists in the same Voronoi cells.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/11ed8798-ffa5-4c89-ad79-eb7abdbad94e)

**This works very well for dense vectors, but not so much for [sparse vectors](https://github.com/facebookresearch/faiss/issues/1922)**. The other common algorithm implemented is **HNSW**, which stands for **Hierarchical Navigable Small Worlds**. It also uses Euclidean distance as a metric but instead of clustering, it is a proximity graph-based approach. There are a lot of nitty-gritty details over here, but we will focus on the main structural components that make up HNSW.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/8bee7f77-096b-4d11-9459-f02a0b197c9f)

The first is what we call as a linked list or a skip list. So on the left image, you will see that as we go from layer 0 to layer 3, we skip more and more intermediate nodes or vertices. We are looking for the nearest neighbor by traversing from left to right and if we overshot, we will move down to the next, to the previous layer. But what if there are just way too many nodes, needing us to build many layers? 
The answer is to introduce hierarchy. So let's look at the top right image. We begin at a predefined entry point and then we traverse through the graph to find the local minimum, where the vector actually is the closest to the query vector.

Now we just went through the vector search strategy. And I want to emphasize that the ability to search for similar vectors is actually not a small feat because it opens up the possibility of our use cases by a ton. We are no longer limited to writing code that is constrained by exact matching rules. In fact, when we write exact matches, we are using filter statements and, as we all know, SQL filter statements are often not very flexible. So that's what we're going to cover next: how do these vector databases or vector storage solutions actually implement filtering?

# Filtering
