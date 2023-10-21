# Embeddings, Vector Databases & Search

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/82fe632c-2fcd-4d26-8e2e-7ac07c78c148)

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
Adding filtering function in vector databases is actually quite hard. And different vector databases also implement this differently as well. There are largely three categories of filtering strategies: you can either do filtering post-query, in-query, or pre-query. But there are also vector databases, which implement their own proprietary filtering algorithms, grounded in one of these as well. So take the example of this query, I can start out by searching for only shoes, which gives me the broader selection choices possible. And then I can make my queries more and more precise by adding adjectives like "black", "black shoes" in the query. But maybe I finally decide I want to buy only Nike shoes. So we can view this brand "Nike" as a brand-specific callout in the metadata. And in fact, for many online retail websites,
we can indeed filter by this metadata, for example, by brands, by styles, by price and etc. 

So a question for you is, would you implement the brand filter prior to the query? What if you don't actually have any Nike shoes in your store? Should you then return no shoes in your results? Let's go through the nuances of each filtering type in the following slides.

## Post-query Filtering
The first is post-query filtering.
Say that now you are trying to find the best Pixar movie that's similar to Frozen. After we identify the top-K nearest neighbors ordered results, we can then apply the Pixar Studio filter. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/088300d7-92ad-494e-88d7-fe4c74bac8f0)

The upside here is that we get to leverage the speed of ANN but the downside is that the number of results is highly unpredictable and maybe there's no such movie that meets your requirements.

## In-query Filtering
The second type of filtering is in-query. This is quite interesting because the algorithm does both ANN and filtering at the same time. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/f9acb986-4b5c-466e-9758-d10654b165b2)

For instance, when you search the movie again that is similar to Frozen but produced by Pixar, all the movie data will have to be converted into vectors. But in the meantime, the studio information is also stored in the system as a scalar field.

So during search, both vector similarity and metadata information will need to be computed. This can put a rather high demand on the system memory because it needs to load both the vector data and the scalar data for filtering and vector search. So perhaps when you shop online and when you have used a lot of filters at once, you might realize that when you add more filters, the website sometimes may take more time to return you the results. 
So if the memory of the system is limited, as you increase the number of filters, you might actually hit out-of-memory issues, abbreviated as OOM, in this slide. But this approach is actually quite suitable for **row-based data** because, in a row storage format, you need to read in all columns in a row at once; as opposed to for columnar storage format that allows you to read in subsets of columns of data at a time.

## Pre-query Filtering
The last type is pre-query filtering. This limits similarity search within a certain scope. What vectors can I actually even consider, you know, after I apply the filter? The downside of this approach is that it doesn't leverage the speed of ANN and all data will have to be filtered in a brute-force manner. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/d1073edb-7371-428c-814d-ebaafc32f7af)

So this is often not as performant as the post-query method or the in-query filtering method because the other two methods could easily leverage the speed of ANN.

# Vector Stores
We are now on to the more practical aspect of how we interface with these vectors. The answer is using **vector stores**. Loosely speaking, when I talk about vector store, it can include vector databases, vector libraries, and also plugins on top of their existing regular databases.
But why do I care about vector stores? Why can't I just use a regular database to store vectors?

## Why are vector database (VDBs) so hot?
Vector stores aren't actually too different from regular databases. Specifically, a vector database is actually just like a regular database. It inherits full-fledged database properties like CRUD, which stands for Create-Read-Update- and Delete. But a **vector database is specialized to store unstructured data as vectors** and in fact, the differentiating capability of vector stores is providing search as a service. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/b2492932-01d3-4f42-a59e-6d70ced5ff1e)

You don't have to implement your own search algorithm. Vector stores provide search functionality for you out of the box. 

## What about vector libraries or plugins? 
Let's talk about libraries first. So vector libraries do create vector indexes for you and as we mentioned a few segments ago, a vector index is a data structure that helps you to conduct efficient vector search.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/6522893b-be63-4232-ba07-89a3148bd087)

So if you don't want to integrate with a new database system, it's actually completely fine to use a vector library that creates this vector index for you. Typically a vector index can contain three different components: 
- the first is an optional pre-processing step that users typically implement on their own, where you may want to normalize your embeddings or reduce the embedding dimensions. The primary step is where an indexing algorithm is actually involved; for example, we talk about **FAISS** and we talk about **HNSW**
- And the last optional post-processing step is where you may actually want to further quantize or hash your vectors to optimize for search speed.

So a vector library like FAISS is often sufficient for small and static data but all vector libraries do not have database properties, so it means that you wouldn't come to expect a vector library to have vector database properties, like the CRUD support, data replication or being able to store the data on disk, or you'll probably just have to wait for the full import to complete before you can query. And it means that it also means that every single time you make changes to the data, the vector index will have to completely rebuild from scratch. 

So whether or not you use a vector database or a vector library really comes down to how often does your data change and whether you need the full-fledged database properties that comes with a vector database or no. 

On the other hand, there are also existing relational databases or search systems that provide you Vector search plugins they typically have fewer metrics or ANN choices but I won't be surprised if you will see a lot more vector search support for these plugins, even in the coming months. 

## Do I need a Vector Database?
So now, let's talk a little bit more about vector databases or not.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/8649733a-7054-4ad6-83f6-f065d141566c)

So let's start by remembering that whether or not you use a vector database, it doesn't affect the speed of your ANN under the hood. The decision comes down to three main things:
- do you have that much data? Typically, we'll only see the need for having a vector database when you have millions or billions of records and how fast do you actually need the query time to be your serving time your latency.
- And lastly, as I mentioned earlier, do you actually need the full fledged database properties? So if your data is mostly static and you don't expect to update your data all that much, then not using a vector
database is often a fine start in that case. You can often just start by using a vector library, but if your data changes quickly, it can be much cheaper to offline complete compute the embeddings first and then store them in a vector database for on-demand query later. This way you can also avoid using an online model to dynamically compute the embeddings and, of course, unsurprisingly, the cons for adding a vector database to your architecture means that you are going to pay for an additional service and you do have one more system to learn, integrate and maintain.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/200c75a9-feb3-4570-9dfc-107e3ce41162)

If you are interested in exploring vector databases, I've provided above some starter comparisons across the popular choices and note that the information here may evolve over time.

# Best Practices
## Do I always need a Vector Store?
In the context of LLMs, whether or not you need a vector store, you know, whether it is a vector database or a library or a plugin on top of your relational database, it all comes down to do you need context augmentation. Vector stores extend LLMs with knowledge and it can provide relevant vector lookup and therefore extend the context. So this can be really helpful to help with factual recall, as we mentioned. And it can also help with the concept called **hallucination** which is an LLM problem that we'll dive into in Module 5. But generally speaking, there are use cases that probably do not need context augmentation to help with factual recall. For example, summarization, your text classification use cases including sentiment analysis, and translation. For these use cases, you probably should feel safe enough to not use vector stores.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/aefa3d8a-0757-47b1-8a6e-a76ab99950ac)

- Vector stores extend LLMs with knowledge
  - The returned relevant documents become the LLM context
  - Context can reduce hallucination (Module 5!)
- Which use cases do not need context augmentation?
  - Summarization
  - Text classification
  - Translation

## How to improve retrieval performance?
How do you improve retrieval performance then, to allow users to get better responses?
At a very high level, there are two different strategies. One is regarding your embedding model selection and the second has to do with how you store your documents. Let's start with embeddings.

Tip one: you should absolutely choose your embedding model wisely. A proxy question that you can ask yourself is: is your embedding model currently trained on similar data as yours? If the answer is yes, then good news, you can keep using the embedding model.

- Embedding model selection
  - Do I have the right embedding model for my data?
  - Do my embeddings capture BOTH my documents and queries?
- Document storage strategy
  - Should I store the whole document as one? Or split it up into chunks?
 
### Tip 1: Choose your embedding model wisely
A proxy question that you can ask yourself is: is your embedding model currently trained on similar data as yours? If the answer is yes, then good news, you can keep using the embedding model.
But if the answer is no, then you have two options over here. First is to look into using another pre-trained embedding model. Or the second is to either train your own embeddings or fine-tune your embeddings based on your data. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/439e4d4d-4a53-4573-bbb9-4ecde17b76ca)

The latter approach over here has been around in the field of NLP for years. It is a very established approach and we used to talk about fine-tuning BERTembeddings all the time before the hype of ChatGPT or chatbots surfaced.

### Tip 2: Ensure embedding space is the same for both queries and documents
Make sure that your embedding space actually captures all of your data, including your user queries as well. For example, if your data is about movies and you ask something about medicine then the search retriever system would definitely have a bad performance.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/b93b9b3d-544b-484f-8e79-ccf8ee6f8a99)

So just always make sure the documents in your vector database actually contain relevant information to your queries. Similarly, use similar models to index your documents and your queries if you want them to have the same embedding space. And the same embedding space is really important if you want relevant results to be returned.

## Chunking Strategy: Should I split my docs?
Now, onto document storage strategy. I'm going to preface all of this with a caveat that how to best store your documents is still not very well defined but I'll share some points for your consideration when it comes to document storage. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/c6a71eb4-9992-4543-9fda-dff9b56a4f55)

We have two choices:
- one is either to store a document as a whole document or we can store a single document by chunks. It means that we are splitting a document up into multiple chunks so each chunk could be a paragraph, could be a section or could just be anything that you arbitrarily define. It means that one document can produce many vectors and your chunking strategy may determine how relevant is the chunks returned to the query
itself but you also need to think about how much context or chunks can you actually fit in within the model's token limit? Do you need to pass this output to the next LLM?

So passing outputs to another LLM is something that we haven't touched upon in this module, but we'll talk about it in Module 3. As an example, if you were to have four documents with two thousand tokens in total, it could be that each chunk has roughly 500 tokens. That will be to split the document even evenly.

### Chunking strategy is use-case specific
But know that chunking strategy is highly use-case specific. In machine learning, we talk about how developing a model is usually an iterative process and you should absolutely also treat chunking strategy as in the same way as well. Experiment with different sizes and different approaches.
How long is your document? Is your document with single sentence or many many sentences? If a chunk is only one sentence, then your embeddings will only focus on specific meaning for that particular sentence. But if your chunk actually captures multiple paragraphs, then your embeddings would capture broader themes of your text.

You can split by headers; you can split by sections; you can split by paragraphs.
But you should also consider the user behaviors as well. Can you anticipate how long the user queries will be? If you have longer queries, then there is a higher chance for the query embeddings to be aligned better with the chunks that are returned. But if you have shorter queries, then they tend to be more precise and maybe having a shorter chunk would actually make sense.

# Chunking best practices are not yet well-defined
It’s still a very new field!
Existing resources:
- [Text Splitters](https://python.langchain.com/en/latest/modules/indexes/text_splitters.html) by LangChain
- [Blog post on semantic search by Vespa](https://blog.vespa.ai/semantic-search-with-multi-vector-indexing/) - light mention of chunking
- [Chunking Strategies by Pinecone](https://www.pinecone.io/learn/chunking-strategies/)

Now say that I choose the wrong embedding model and my chunking strategy was not good, can we actually add some guard rails to prevent silent failures or undesired performance?
So for users, it will be helpful for you to actually include explicit instructions in the prompts. As we discussed in Module 1, where you can tell the model not to make things up if it doesn't know the answer. So this can help you to actually know where the model limitation is rather than relying on unreliable outputs.

But for software engineers, there are a few things that you can consider
- First is to maybe add a failover logic.
  - If the distance-X exceeds threshold, then maybe you have to show a generic list of responses, rather showing nothing. So going back to the Nike example, if there are no Nike shoes, return then probably you can show a generic list of the most popular shoes that users can buy.
- In terms of toxicity or discrimination or exclusion, you can also add a basic toxicity classification model on top.
  - Prevent users from actually submitting offensive inputs.
  - Discard offensive content to avoid training or saving to VDB.

In 2016, there is this chatbot released by Microsoft called Tay that actually became a really racist chatbot because users start submitting racist remarks. So by having some guardrail model on top will help prevent a chatbot from functioning differently as you expect.

And you can also choose to discard all the offensive content to avoid retraining or fine-tuning on this offensive content. And lastly, you should also can think about consider configuring your vector database to actually timeout if a query takes too long to return a response. Maybe this indicates that there are actually no similar vectors found.

# Module Summary
- Vector stores are useful when you need context augmentation.
- Vector search is all about calculating vector similarities or distances.
- A vector database is a regular database with out-of-the-box search capabilities.
- Vector databases are useful if you need database properties, have big data, and need low latency.
- Select the right embedding model for your data.
- Iterate upon document splitting/chunking strategy

# Module 2 - Resources
Research papers on increasing context length limitation
- [Pope et al 2022](https://arxiv.org/abs/2211.05102)
- [Fu et al 2023](https://arxiv.org/abs/2212.14052)

Industry examples on using vector databases
- FarFetch
  - [FarFetch: Powering AI With Vector Databases: A Benchmark - Part I](https://www.farfetchtechblog.com/en/blog/post/powering-ai-with-vector-databases-a-benchmark-part-i/)
  - [FarFetch: Powering AI with Vector Databases: A Benchmark - Part 2](https://www.farfetchtechblog.com/en/blog/post/powering-ai-with-vector-databases-a-benchmark-part-ii/)
  - [FarFetch: Multimodal Search and Browsing in the FARFETCH Product Catalogue - A primer for conversational search](https://www.farfetchtechblog.com/en/blog/post/multimodal-search-and-browsing-in-the-farfetch-product-catalogue-a-primer-for-conversational-search/)
- [Spotify: Introducing Natural Language Search for Podcast Episodes](https://engineering.atspotify.com/2022/03/introducing-natural-language-search-for-podcast-episodes/)
- [Vector Database Use Cases compiled by Qdrant](https://qdrant.tech/use-cases/)

Vector indexing strategies 
- Hierarchical Navigable Small Worlds (HNSW)
  - [Malkov and Yashunin 2018](https://arxiv.org/abs/1603.09320)
- Facebook AI Similarity Search (FAISS)
  - [Meta AI Blog](https://ai.facebook.com/tools/faiss/)
- Product quantization
  - [PQ for Similarity Search by Peggy Chang](https://towardsdatascience.com/product-quantization-for-similarity-search-2f1f67c5fddd) 

Cosine similarity and L2 Euclidean distance 
- [Cosine and L2 are functionally the same when applied on normalized embeddings](https://stats.stackexchange.com/questions/146221/is-cosine-similarity-identical-to-l2-normalized-euclidean-distance)

Filtering methods
- [Filtering: The Missing WHERE Clause in Vector Search by Pinecone](https://www.pinecone.io/learn/vector-search-filtering/)

Chunking strategies
- [Chunking Strategies for LLM applications by Pinecone](https://www.pinecone.io/learn/chunking-strategies/)
- [Semantic Search with Multi-Vector Indexing by Vespa](https://blog.vespa.ai/semantic-search-with-multi-vector-indexing/) 

Other general reading
- [Vector Library vs Vector Database by Weaviate](https://weaviate.io/blog/vector-library-vs-vector-database)
- [Not All Vector Databases Are Made Equal by Dmitry Kan](https://towardsdatascience.com/milvus-pinecone-vespa-weaviate-vald-gsi-what-unites-these-buzz-words-and-what-makes-each-9c65a3bd0696)
- [Open Source Vector Database Comparison by Zilliz](https://zilliz.com/comparison)
- [Do you actually need a vector database? by Ethan Rosenthal](https://www.ethanrosenthal.com/2023/04/10/nn-vs-ann/)
