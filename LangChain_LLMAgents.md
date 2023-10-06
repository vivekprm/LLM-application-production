Lecture by Harrison Chase, creator of LangChain.

# Overview of LLMChains & Agents
The core idea of this is using an llm as the reasoning engine. So we'll use the llm to determine how to interact with other sources of data and computation, but crucially we will use those sources of data and computation for their knowledge rather than using the knowledge that's present in the llm.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/6d79cfb8-aec7-4f97-89db-0da4ee7d1ce9)

## Retrieval Augmented Generation Chatbot
We'll use the llm more as a reasonable engine. And so the canonical example of this is **retrieval augmented generation**. And we can think of this as a chatbot
that can answer questions about particular documents so documents that are proprietary to you that the language model GPT-3 or whatever has not been trained on.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/bd7ba380-84ab-4b6f-8bbf-7c956e9d378b)

And so the general flow for this type of application is first you'll have the chat history which you'll condense into a standalone question. And this is important because you'll use that question to retrieve the relevant documents. If you don't do this convention step and you only use the final chat message in the chat history it could be referencing things prior that you won't know about. So this so this **condensing step is really important**.

So you do that then you look up relevant documents and you get back a list of relevant documents and then you pass this to a language model to generate the final answer and so this will take in the original question the chat history the retrieve document, which here now which the prompt will generally ground the the response in and it will generate a final answer. And so here you can see that we're using the language model as a reasoning engine which takes in a question and some some data and then
reasons about an answer relative to the data that's been provided. But we're basing everything, we're we're grounding everything in that data that we've retrieved we're not using the knowledge present in the language model itself for that.

# Why is evaluation hard: Lack of Data
So why is evaluation of these types of applications hard? First is due to a lack of data. So unlike traditional problems in machine learning you generally don't start with a data set. You generally start with an idea, an problem or an application that you want to build, and then you start building and then you need to evaluate it and you don't have you don't have the traditional machine learning training data set to start from.

It's also unclear for this what the data set would even be so for a lot of these questions. If we start to think of some of the more complex ones they could be constantly changing based on the day. So a lot of question answering applications have to do with up-to-date data, so it's unclear if there's even a ground truth answer that remains constant over time so it's really challenging to gather data.
