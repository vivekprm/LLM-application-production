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
