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

## Why is evaluation hard: Lack of Data
So why is evaluation of these types of applications hard? First is due to a lack of data. So unlike traditional problems in machine learning you generally don't start with a data set. You generally start with an idea, an problem or an application that you want to build, and then you start building and then you need to evaluate it and you don't have you don't have the traditional machine learning training data set to start from.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/8ffde23f-646f-4c9d-9165-da98496cf5bb)

It's also unclear for this what the data set would even be so for a lot of these questions. If we start to think of some of the more complex ones they could be constantly changing based on the day. So a lot of question answering applications have to do with up-to-date data, so it's unclear if there's even a ground truth answer that remains constant over time so it's really challenging to gather data.

## Why is evaluation hard: Lack of Metrics
The other issue is a lack of metrics so it's hard to even evaluate with your eyes. So a lot of these applications if you think about the chain before there's three different steps that could be going wrong at any of those intermediate steps and you might not necessarily know which one and where.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/9ab621b7-dad7-4aeb-8065-69bcc1c9daa1)

And then it's also tough to evaluate quantitatively, with metrics looking at the at the final answer. So in the question or in the example above of a question answering chatbot, the final answer that the chatbot gives is freeform text. And so it might have a particular fact in it and that's the thing that we should be evaluating but there's a lot of conversational text around it as well so you can't do things like exact match on that text and you need to do some sort of of more advancing so which we'll talk about later.

## Potential Solutions - Best Practices
So some potential solutions here, that are emerging as best practices. First around the lack of data. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/9606eb10-e155-43b5-b6f7-27c1fd28084c)

There's two things here that that people tend to do. One is generate data sets ahead of time and so you can generate these programmatically and often times a
language model is part of that programmatic generation of these things.

So in the example of question answering over a document if you want to generate a test set for it one thing one chain that we commonly used to do that is a chain that splits the document up into chunks and then for each chunk you ask the language model to generate a question and answer pair and that becomes kind of your test set for
this application.

The other thing that is commonly done to gather this data is just accumulated over time, so if you have a application running in production you can keep track of the inputs and outputs and you can start to eventually add these to a data set and grow that data set over time.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/c25dff0b-246f-4a41-8e10-341e434903aa)

For lack of metrics, the first and most important thing is just to make it really easy to inspect visually what's going on. So what are the inputs and outputs at each step in the chain. What are the inputs when it finally goes to the language model and it finally comes back. This is really important for understanding if the application is messing up due to a lack of retrieving the right information or it's retrieved the right information it's just not synthesizing it properly.

Another thing that we can do is use a language model to judge whether the final answer is correct or not so this will handle the cases where there's a fact but it's surrounded by a bunch of conversational texts and so we'll use the language model to take a natural language answer and then a natural language ground truth label and compare whether they're semantically equivalent.

And then the final thing that we can do here is we can just gather feedback, directly or indirectly, from users and so I'll talk a little bit more about this during an online evaluation.

## Offline Evaluation
But first, offline evaluation. So the main way to evaluate chains and agents in an offline manner, is:
- Create a data set of test points to run against.
- You then run the chain or agent against them
- You then visually inspect them so you can you can look at the inputs and outputs at overall, at each step, see how it's doing, and then that's not really scalable of course.
- And so then the next thing to do is use a language model to auto-grade them and you can either fully trust this auto-grade. So you could have it assigned correct or incorrect to each run and then just average that, or you could just use that as a way to guide your eyes to the data points that might be most correct or incorrect and
then do another visual inspection on top of there.

An offline evaluation is usually used before the model goes in to production so this can be right after you've finished developing it and you're doing a final test to see whether it's ready to go in production.

## Online Evaluation
Online evaluation, on the other hand, is after the model's been deployed. So now it's running in production you're serving users and you want to make sure that it's continuing to do a good job. And so the main method of doing this is:
Gathering feedback on each incoming data point so there's ways of 
- Gathering this feedback directly. So you could add a thumbs up or a thumbs down button to your application and then you can click on it and you can you can track this
feedback over time. And if you release a new version of the application or if the model starts doing poorly, for whatever reason, you can hopefully notice a downtrend in this feedback. And you can use that as a mechanism to know that you need to fix your model.
- There's also other forms of feedback that can be more indirect. So if you're serving up, you know, relevant links if someone clicks on a link that probably means you did a good job if they did not click then that means that the model might not be doing a good job. And so you can use this as an indirect measure of feedback.

And both of these, the idea is to track these over time, and you can start to notice when the model performance starts to degrade.
That is pretty much it!

## The Future Of Evvaluation in LLM Chains
Evaluation of these applications is a new and exciting topic. These applications are just starting to kind of like go in production so there's just starting to emerge kind of like a set of best practices around them and I'm sure we'll continue to add and grow our knowledge about evaluation as more and more go into production so I'm personally really excited to see what develops over the next few weeks, months, and years of running these llm applications in production.
