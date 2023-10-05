# Multi-stage Reasoning
This is a really exciting capability that people are developing with LLMs basically building more complicated applications out of individual calls to language models. So in this module we're going to start off by talking more about prompting and how to how to build more sophisticated prompts that get a model to reason in some way to to produce better answers. 

Typically for building an application the llm will just be one part of the entire workflow of this end-to-end application that we're designing so we need to think about how we can link together llms and other pieces of our code so that they work seamlessly together and then if we need to maybe swap in at one llm for another it doesn't break the entire end-to-end system.

Let's consider an example where we wanted to summarize and get the sentiment of a particular article which is not an outlandish or ridiculous task if we think about how we might approach this with just one large language model we could give it a bunch of different articles ask it to summarize it and then get the sentiment. That's quite a few tasks for it to deal with in one go.

A better strategy might be to take one article at a time, pass it through a summarization large language model and then the output of that summary could then be given to a sentiment analysis large language model.

Now we could think that conceptually we have all the tools we need to do this but how do we actually do this in a programmatic fashion?
If we think about the issues of having one large language model do all of this we've run into all kinds of issues with needing a very large, llm to be able to both summarize and perform sentiment analysis as that's quite a complicated task to do in one go and we also need to worry about the fact that if we're giving all of the articles to the llm in our prompt will very quickly overwhelm the input sequence length that the llm is built for.

So what we need to do is break up every article into a single article pass them through one by one, collect the output of our summary llm and give those summaries as the input to the sentiment llm that we're going to create.

This framework then becomes a reusable tool so that we can keep passing it in new articles and it can generate piece by piece, the different steps of this application.

So let's focus on that first task instead of passing all of the articles in as a giant prompt we're going to pass them in one by one.
So we want to make sure that we have a systematic way to abstract out each article so that we can input them almost as a variable.

## Prompt Engineering
A well-written prompt can elicit a good response from a large language model and a poorly written prompt can sometimes leave some of the performance that large language
model has to provide, on the table. We want to make sure that we're using the best practices for using our prompts and in this particular situation we need a systematic approach so that we can summarize all of our articles one by one.

A well-written prompt saves a lot of hassle, and can be shared and modularized across the team and across the community.
Let's take a look at some example code of how we might build up our prompt for this particular use case. To recall, we're going to use a summarization large language model for an article that we're going to give that eventually we'll use to get the sentiment from.

```py
# Example template for article summary
# The input text will be the variable {article}
summary_prompt_template = """
Summarize the following article, paying close attention to emotive phrases: {article} Summary: """
```

{article} is the variable in the prompt template.

We're going to start with a summary prompt template as the article needs to be summarized and we're going to build this template in step by step.
We give the large language model a task description, so we're telling it to summarize the following article paying close attention to emotive phrases. That second clause is there because we want to make sure that we're looking for sentiment, maybe less than factual information.

We see the curly braces there define the variable that we'll be inputting later on. That input will be the text of the article that we want to summarize.

Finally, we give the summary and colon so that it knows that it has to start producing the summary as the output.
This implies that we're using a generative model instead of the classification model that we also discussed in the primer.

The next step then would be to create a prompt template. We're using some of the syntax here from **LangChain**, but if you have a different prompt library that you're working with you'll see something similar.

```py
# Example template for summarization
# The input text will be the variable {article}
summary_prompt_template = """
Summarize the following article, paying close attention to emotive phrases: {article} Summary: """
#############################################################################################
# Now, construct an engineered prompt that takes two parameters: template and a list of input variables (article)
summary_prompt = PromptTemplate(template=summary_prompt_template, input_variables=["article"])

#############################################################################################
# To create an instance of this prompt with a specific article, we pass the article as an argument.
summary_prompt(article=my_article)
# Loop through all articles
for next_article in articles:
   next_prompt = summary_prompt(article=next_article)
   summary = llm(next_prompt)
```

We'll give the prompt template that we just created a moment ago and we'll define the input variables we'll call that article so that it's consistent with the one that we put in the template. We then have an instance of our prompt template using the summary prompt to store that.

And then finally if we have an article that we'll call my underscore article, we'll set that to be the input variable for this new summary prompt. This summary prompt now has an instance of the article plus the prompt template that we created before. We could then pass this to our summary large language model and produce a summarization of that article which would hopefully pay close attention to emotive phrases in that article.

We could loop over all of the articles that we have and create summaries for each of them.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/ef8d22e9-44bf-4c09-927b-e28a9089b6bd)

So this solves one part of this two-stage problem where initially we needed to give articles one by one and give that to a summary large language model.

But now we need to think about the output of that summary large language model so that it can be the input to a sentiment analysis large language model.
So you can think of what we've done up to now is chaining a prompt to a large language model. This was done using prompt templates and now we need to chain one large language model with another large language model and so we'll delve into the world of **llm chains**.

## LLM Chains
Now let's start with one of the most exciting areas of large language models and, that's llm chains where we can link together not just one llm with another but even a variety of different tools.
The idea of llm chains really started to come into popularity at the end of 2022 with the release of the **LangChain** library.

Let's go back to our example where we've finished taking in article, summarizing them and creating a prompt template to do that summary. We now need to create another prompt template so that we can take our sentiment analysis and put that into our workflow.

So we're going to create a new sentiment prompt template like we did for the summarization, and we're going to say evaluate the sentiment of the following summary and
then pass in that summary.

```py
# Firstly let’s create our two llms
summary_llm = summarize()
sentiment_llm = sentiment()
# We will also need another prompt template like before, a new sentiment prompt
sentiment_prompt_template = """
Evaluate the sentiment of the following summary: {summary}
Sentiment: """
# As before we create our prompt using this template
sentiment_prompt = PromptTemplate(template=sentiment_prompt_template, input_variable=["summary"])
```

The llm is then going to be requested to produce the sentiment so this is very similar to what we saw before we're just finishing the loop of this problem. So we
have our two large language models they could come from the same provider, they might be different ones, depending on how we want to leverage the resources we
have at hand we might use a large language model that's fine-tuned for summarization and one that's fine-tuned for sentiment.

So now we have the prompt that will take in as an input the summary of the article that we started with. So we're going to take the output of the first large language model as the input to the prompt for our next large language model.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/da8300d6-d693-4688-9668-4f6da17cb87b)

So if we look at how we connect these two together, we end up with three different chains. Firstly we have our workflow chain that connects all of the pieces together,
we then have inside our workflow chain two smaller chains: the summary chain, which we saw previously connects the prompt and the article data to the large language model that we use for summarization.

The sentiment chain connects the output of the summary chain as the input to the prompt for our sentiment analysis large language model and then the output of the sentiment chain becomes the output of the workflow chain for the sentiment of article 1.
