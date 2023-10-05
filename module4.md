# Fine-tuning and Evaluating LLMs
We've already seen some amazing things that we can do with large language models, from using hugging face to answer almost any problem in natural language processing to the vector databases that we saw in Module 2 and then the amazing tools that we can use and apply with LangChain. It seems like large language models and the applications surrounding them give us flexibility to build almost anything that we want.

But what if there's an application that you think a large language model that exists isn't quite suited for? Maybe it needs customized dataset to work with or maybe you're just not sure how to build your application in the most efficient manner. The goal of this module is to show how we can take different types of large language models, apply them to build a specific type of application, and go through the process of fine tuning if we think that we just need to create something that's special
for us.

# A Typical LLM Release
![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/d8170a77-5ed8-490e-87e0-f2eff342161b)

So let's talk about a typical large language model release. We're going to consider the open source large language model releases that tend to occur almost weekly at this point.
Usually they're released in a number of different sizes. So we have a foundation model which is a model that's just trained to predict the next word on all of the text that it's seen so far. These typically come in a base model size and then a smaller and a larger version. The size here is the number of parameters or the amount of gigabytes that it might take up on your storage or RAM.

It could also be released with different sequence lengths, so these are the amount of input tokens that we can give to a model in a single pass.

Newer techniques, and we'll cover some of these in Course 2, allow large language models to expand the amount of sequence length up to tens of thousands whereas typically we're limited to around four thousand. We might also see that a large language model is released with some pre-fine-tuned versions along with the base model that is built just to predict the next word for a generative model.

We might also see a chat based model that is released with this foundation release that's been trained to converse in a more human-like interaction and we might also see an instruction-based model which is slightly different to the chat based model in that it's specifically designed just to respond to tasks that it has been given.

## So as a developer you might wonder which direction do you take, which one do you choose?
In every situation we're going to have to balance accuracy, which tend to favor larger models as the **larger models typically will give better performance as they've seen more training data and have a larger parameter count to solve different problems**.
Speed, so the speed for inference is going to be an important factor and **usually smaller models, because they take up less space and there are less calculations involved, are much faster at inference than larger models**. And then the task specific performance is also something that we need to take into account. 

**Larger models, while they might have a broad knowledge set, might not be as efficient or task-specific performance values might be different for these just bland large models rather than a fine-tuned model on a specific use case**.

# Applying Foundation LLMs
Our application is going to be a news app that summarizes the daily news articles and presents them as riddles for the users to read through to try and decipher what's going on.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/2fc59043-351b-4b78-915f-c5e105666ac2)

Let's think about how we might build this using the tools that we have available.

## Potential LLM Pipelines
We have a number of different potential llm pipelines.

![Uploading image.png…]()

Let's consider what we already have we'll have an application programming interface or an API connected to some sort of news outlet so that we can get daily articles of what's going on around the world. And we might have some pre-made examples written by either us or people that we know are good at writing riddles from news articles. We
don't have too many samples but we've got enough to maybe do some few shot learning later on.

If we think about the llms that were released by the community we could do some few-shot learning with that open source llm. We could use the instruction-following
llm and do some zero-shot learning there.

We could use an llm as a service paid option, or we could go down the build our own path. Ultimately we just want to end up with a way to take the news articles and
present that as an application interface for our users.


