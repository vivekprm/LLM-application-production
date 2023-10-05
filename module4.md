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

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/e4b7b767-c2c4-4109-820c-1359c0f4a668)

Let's consider what we already have we'll have an application programming interface or an API connected to some sort of news outlet so that we can get daily articles of what's going on around the world. And we might have some pre-made examples written by either us or people that we know are good at writing riddles from news articles. We
don't have too many samples but we've got enough to maybe do some few shot learning later on.

If we think about the llms that were released by the community we could do some few-shot learning with that open source llm. We could use the instruction-following
llm and do some zero-shot learning there.

We could use an llm as a service paid option, or we could go down the build our own path. Ultimately we just want to end up with a way to take the news articles and
present that as an application interface for our users.

# Fine-Tuning: Few-shot learning
Let's take a look at the first way that we could tackle this problem with few-shot learning.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/463e9be3-1ee5-49df-bc8b-9cac3b98af46)

Just to recall we have the news API that allows us to gather articles from our news outlet, and we have some pre-made examples of other articles that have been turned into riddles so we could potentially have a look at how feasible the few-shot learning approach might be.

If you think about the pros and cons of few shot learning: it's certainly quick to develop as we have all the data that we need and we just apply our llm as is with just a specified prompt.

For performance sake we'll need probably a larger model as the few examples that we have will be needed so that we can get good performance with a lower number of examples or with a smaller model we tend not to get as good a performance with few shot learning.

## Pros and cons of Few-shot Learning
![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/c9af37d1-1916-41dc-80f8-d6d82455c031)

## Riddle me this: Few-shot Learning version
Let’s build the app with few shot learning and the new LLM. Our new articles are long, and in addition to summarization, the LLM needs to reframe the output as a riddle.

```py
prompt = (
"""For each article, summarize and create a riddle from the summary:
[Article 1]: "Residents were awoken to the surprise..."
[Summary Riddle 1]: "In houses they stay, the peop... "
###
[Article 2]: "Gas prices reached an all time ..."
[Summary Riddle 1]: "Far you will drive, to find..."
###
...
###
[Article n]: {article}
[Summary Riddle n]:""")
```

So let's build our prompt we're going to tell the large language model that it needs to summarize and create a riddle from that summary, now if you think back to what we did in Module 3 for LangChain we might split this up into two different steps. But for now we'll just consider this to be a single approach just to simplify the prompt here.
We'll have all of our articles and the summary riddles as examples that we give in our prompt. 

And then the final part of our prompt will be the article that we have and then an empty space for the summary riddle so that the llm knows to begin by producing the summary riddle for that article.
We need to keep in mind though, that **for these kinds of applications we'll probably need a very long input sequence model**. That might be a very large version of
the model or it might be one that's actually quite difficult to get hold of as long input sequence models are only starting to be released.

Typically for few-shot learning the larger the base or foundation model that you use the better the performance that you have and so that will need to be taken into account as well.

# Fine-Tuning: Instruction-following LLMs
Another way that we could potentially use some of these open sourced llms, and that's by using the pre-fine-tuned instruction following version of the llm.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/382cc798-1f56-4238-9765-a910964db234)

In this situation we're going to assume that we actually don't have any pre-made examples. If we do have pre-made examples then we could follow a very similar path to what we did in the few-shot learning example with just the foundation model.

Depending on how the instruction model is constructed, depending on how it was trained using the foundation model or the instruction following model with few short examples might give similar results or the instruction model might be better or worse depending on how it was trained. In this example we want to go through the scenario where you actually don't have any pre-made examples to work with in that situation we'll actually need to utilize zero shot learning where we just describe the task and then give the article for the model to summarize.

So here we just have the news API, we have our instruction-following llm that's been released by the open source community, and we want to create our app.
So let's think about some of the pros and cons.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/ba4de306-ca76-4a35-b5ed-070777628aff)

## Riddle me this: Instruction-following version
The way that we'd implement something like this would be a very small prompt, much smaller prompt than the one from the few-shot learning where we just described the tasks that we needed to solve and then we give it the article as the input variable and ask it to produce the output.

```py
prompt = (
"""For the article below, summarize and create a
riddle from the summary:
[Article n]: {article}
[Summary Riddle n]:""")
```

Depending on how well it is trained, as I said, we might get different results. We may get fortunate that this is already ready to go for us and so we wouldn't need to do any further options however it's possible that this might not work best for us so maybe we need to look at another option.

# Fine-Tuning: LLMs-as-a-Service
Let's look at a slightly different approach where we actually utilize a proprietary or llm as a service offering to solve our application development problem.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/4f4d54b3-2132-42e6-b16b-d4fe5c0e4d0b)

In this scenario we're going to assume that we don't have any pre-made examples to send along with our news API results. But you could imagine that if we wanted to include some pre-made examples for few-shot learning with our paid llm as a service we could do that too.

The goal for this though is to really look at how incorporating an llm as a service would fit into the workflow for your application.

## Pros & Cons of LLMs-as-a-Service
![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/e63daabb-01cb-4196-824d-355c4a1dd646)

## Riddle me this: LLM-as-a-Service version
So let's look at how we might implement this it's very straightforward.
Especially relative to some of the other ways of building this llm based application really we just described our prompt as the tokens that we're going to send to our API we'll send it along with some kind of API key that we've attached to some credit card or other funding body and then we receive that response from the servers of the API provider.

```py
prompt = (
"""For the article below, summarize and create a
riddle from the summary:
[Article n]: {article}
[Summary Riddle n]:""")

response = LLM_API(prompt(article),api_key="sk-@sjr...")
```

so this is really the lowest effort and we tend to get very good performance from a closed source providers of llms at this stage in the sort of development cycle of large language models in general the proprietary software so far has outperformed the open source community.

Though there's certainly a race to improve the performance of the open source community as well so let's see. **If none of these situations really fit for us even the closed source proprietary versions, perhaps they're not as good as we really need. Maybe we really do need to fine-tune one of these large language models**. 

# Fine-Tuning: DIY
Let's say we've tried all of the other llm offerings and none of them quite give us the results we want. Maybe they just haven't been trained on data that's specific enough for what we want to use in our application.
In this situation we're going to need to try and do it ourselves.

Let's assume that we still have a good number of pre-made examples that show the article being summarized and transformed into a riddle and we also have our news API connections so that we can get as much inference data as we need.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/fbbe49c7-ea8e-4ee4-b8ea-637f95f5c19f)

So when we come to building our own we have two options we can build our own base model foundation model from scratch which would include constructing the architecture for the base foundation model, we would need to gather a data set that encompasses a huge amount of training data sources and then we would fine-tune that model once
that's been trained or we could take an existing model and fine-tune it on just the data set that we have available right now.

Almost never will we go down that first path of training a foundation model from scratch. It requires the resources of a very large company to do it properly and cost efficiently in most cases really only a dozen or so companies across the world are doing this and making them available in open source. It's just such a time commitment, cost commitment, and resource commitment that it's really infeasible and unnecessary for almost everyone else to do.

But let's go down the fine tuning and existing model path as there's a lot of interesting nuances that we need to keep track of.

## Pros and cons of fine-tuning an existing LLM
![Uploading image.png…]()

