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

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/ee199c47-4a2f-4824-8386-02b02b661c6c)

So when we come to building our own we have two options we can build our own base model foundation model from scratch which would include constructing the architecture for the base foundation model, we would need to gather a data set that encompasses a huge amount of training data sources and then we would fine-tune that model once
that's been trained or we could take an existing model and fine-tune it on just the data set that we have available right now.

Almost never will we go down that first path of training a foundation model from scratch. It requires the resources of a very large company to do it properly and cost efficiently in most cases really only a dozen or so companies across the world are doing this and making them available in open source. It's just such a time commitment, cost commitment, and resource commitment that it's really infeasible and unnecessary for almost everyone else to do.

But let's go down the fine tuning and existing model path as there's a lot of interesting nuances that we need to keep track of.

## Pros and cons of fine-tuning an existing LLM
![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/fbbe49c7-ea8e-4ee4-b8ea-637f95f5c19f)

## Riddle me this: fine-tuning version
The way of fine-tuning instruction following models has really come into its own in the last few months as we've seen some really fantastic results from Stanford University and others working on projects like [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) and [Dolly V1](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html).

Depending on the amount and quality of data we already have, we can do one of the following:
- Self-instruct ([Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) and [Dolly v1](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html))
  - Use another LLM to generate synthetic data samples for data augmentation.
- High-quality fine-tune ([Dolly v2](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm))
  - Go straight to fine tuning, if data size and quality is satisfactory.

However these are somewhat held back as they used proprietary data sets or datasets at least that had commercially restrictive licenses.
In the last few weeks a new version of Dolly, Dolly V2 was released by Databricks using an open source data set.

# Dolly
Let's talk about Dolly, one of the innovations of 2023 that really opened up a new paradigm in large language modeling.

Dolly is a 12 billion parameter model based on the Eleutha AI pythia 12 billion parameter model. Dolly is an instruction-following large language model which means that we can ask it to do specific tasks and it responds in the way that it was trained.

Dolly is special in that it represents an approach that has been neglected or at least not investigated fully by the open source community. However we've seen now a huge momentum shift into this kind of direction in the early months of 2023.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/fc640c71-1368-465e-948c-4890c9102327)

Dolly was built by first taking an open sourced pre-trained foundation model, the pythia 12 billion parameter model which was trained on [The Pile](https://pile.eleuther.ai/) a data set comprised of a number of different open source data sets.

And then for Dolly we fine-tune the model on these [databricks-dolly-15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k) data set. This data set is probably the real core of what makes Dolly special.

The eleutherAI 12 billion parameter model is an open source model that we can use however we wish however, it's not fine-tuned in any particular way. If we do want to fine tune it we need a data set of high quality that we can use so that it elicits responses that's useful for us.

The databricks-dolly-15K dataset was produced by the employees of Databricks and contains within it pairs of instructions and responses of high quality intellectual tasks. The special source behind Dolly is the fact that this databricks-dolly-15K dataset was released by the Databricks owners to be completely open and commercially usable.
This is different from all of the other attempts prior to this as they had some kind of licensing issue. Dolly itself is not a state-of-the-art model, it's just an approach that shows that you can take a model such as the eleuutherAI open source model combine it with a high quality but open source data set and produce something
that is commercially viable.

Many new approaches are now taking place to combine this dolly 15K data set with even newer open source architectures. 

## Where did Dolly come from?
The idea for Dolly came from the [Stanford Alpaca project](https://crfm.stanford.edu/2023/03/13/alpaca.html). In this project they used 175 instructions that they created themselves and gave this to the OpenAI model text-davinci-003 to create synthetic and new versions of these tasks.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/89fd2fbb-9f9c-4eeb-afb8-6fd59c5132dd)

After going through a process of trial and error they ended up with about 52000 high quality, instruction following examples. They combined this with meta's Llama 7 billion parameter model which was released not long ago with a high quality this high quality data set meant that they produced a very capable model.

However this model was restricted with the licensing involved both with the Llama 7 billion parameter model and with the fact that they were using OpenAI to produce more training data. This held back the Stanford Alpaca model from being used in the commercial setting.

This gave a hint though that we could actually use small models with high quality data sets to replicate the kinds of performance we're seeing from these larger models.

**Moving forward now Sam Altman the CEO of OpenAI expressed his sense that we're at the end of this error of chasing larger and larger large language models the focus for 2023 and beyond seems to be now the age of small llms and applying them in different use cases.**

We've moved from this broad approach where we try to build a master of everything and now trying to create it so that we have fine-tuned bespoke models for different types of tasks. Where this will go we're not sure but it's exciting to see where this field keeps evolving and moving into.

# Evaluating LLMs
If we decide to fine-tune our model we really need to understand how to evaluate the model for its performance. Intuitively understanding how well a large language model performs is really quite difficult to verbalize or to come up with a consistent definition.

## Training Loss/Validation Scores
Whilst retraining of course we do look at things like the loss or the validation scores as we go through the training process as these are still deep learning models and they try to optimize some sort of loss function.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/3b754b90-3d6d-4061-82ad-ceb15ce061db)

But for a good large language model what does the loss really tell us? Nothing really, not really any of the traditional metrics you would see for some kind of binary classifier. If you remember what a large language model does, it's really just producing a probability distribution over the entire vocabulary of tokens and selecting which one it thinks is the right answer.

How that really relates to whether or not we're getting a good conversational agent or picking a good riddle for our summaries it's hard to see how those are connected.

## Perplexity
One way that we can improve on just whether or not I got the right answer is looking at its perplexity.

Perplexity is really just how the spread of the probability distribution is over the tokens that its trying to predict. If we have a very sharp probability distribution of our tokens that means it has a very low perplexity and it knows then it's very confident that is the token it should be picking. Whether or not that token is correct or not depends on the accuracy so really what **a good language model will have is high accuracy and very low perplexity**. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/ecac983d-08ea-4962-b2e6-aa1cce5f36d9)

It'll know which word should come next and it'll be correct in picking that word. 

## More than perplexity
Perplexity though is really not the end of the story either, even though we're confident and correct about picking the next word that doesn't necessarily mean that we're getting a good quality of the result they're getting we don't have any context for the rest of the words that is picking in that sentence if it's choosing the same word again and again maybe that has a high value of accuracy and a high value or a low value of perplexity but if we're doing say translation or summarization that's probably going to be complete nonsense.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/14266b25-fdab-40d8-9ffd-a88a33bc1486)

So what we need to look at is task specific evaluation metrics.

## Task-specific Evaluations
### BLEU for translation
For translation we can use the BLEU metric which evaluates how well our output compares to reference samples of the translations that we want to produce.
In BLEU we calculate how many unigrams, those are single words, appear in the output compared to our reference. We also look at how the bigrams, trigrams and quadgrams appear in the output from our model compared to our reference samples.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/c8ec0de8-68c3-4a0a-a20b-987e736ab95a)

BLEU then combines all of these and creates the geometric mean of the uni,bi tri, and quad grams and gives the total value for the BLEU metric.

A unigram in this situation is a single word so if a word appears, say, six times and it's the exact same word it will actually have a very high value for the unigram score in BLEU. However when we extend this to the bigram and trigram case we'll see that the values quickly drop off to zero. If we have a very good translation we'll see that it matches very well the references and so we'll get a high value for the BLEU score.

### ROUGE for summarization
Likewise for summarizations we can look at the ROUGE score.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/9a2bf7d5-fed8-4bd3-b21d-2cefdd905bd0)

ROUGE is quite similar to BLEU in how it matches the reference samples to the outputs given by the summarization model.

In this situation however it also takes into account the length of the summary so that we have as short as summary as possible. If we have quite a verbose summarization but it still contains many of the words in our reference sample that does so-so but not fantastically.

ROUGE then looks for situations where both words are common in both the sample and the output but also that the output is as small as possible relative to the
reference samples. 

### Benchmarks on datasets: SQuAD - Stanford Question Answering Dataset - reading comprehension
But what if we want to do something where we're not using just our data. What if we want to benchmark our model compared to other models. We might not have the same datasets that they have and so that's why the community has also produced benchmark datasets so that we can evaluate our models and compare that to
the others in the community.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/0c54aae3-847a-45b3-b8cd-d007af0a5afa)

**SQuAD the Stanford Question and Answering dataset** is a very commonly used toolkit which contains a number of different data sets that we can use for our models to compare different llms as they've been fine-tuned.

### Evaluation metrics at the cutting edge
Finally some of the more cutting edge evaluation metrics focus on things like alignment. And that's how well if we give say an instruction following llm a particular task does it give us one: a relevant result based on the input that we asked it.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/d6cdeefd-de54-4920-b0fa-f36437986ecb)

Does it give a hallucination which we'll look more at in the next module and is it harmless? Is there a measure of toxicity or profanity in the response that we might
want to reduce depending on the particular use case. Different evaluation metrics are used by researchers and even more becoming produced day by day the problem of alignment though is still a very critical component in modern llm research.

# Module 4 Resources
Fine-tuned models
- [HF leaderboard ](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [MPT-7B](https://www.mosaicml.com/blog/mpt-7b)
- [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)
- [DeepSpeed on Databricks](https://www.databricks.com/blog/2023/03/20/fine-tuning-large-language-models-hugging-face-and-deepspeed.html)

Databricks’ Dolly
- [Dolly v1 blog](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html)
- [Dolly v2 blog](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
- [Dolly on Hugging Face](https://huggingface.co/databricks/dolly-v2-12b)
- [Build your own Dolly](https://www.databricks.com/resources/webinar/build-your-own-large-language-model-dolly)

Evaluation and Alignment in LLMs
- [HONEST](https://huggingface.co/spaces/evaluate-measurement/honest)
- [LangChain Evaluate](https://docs.langchain.com/docs/use-cases/evaluation)
- [OpenAI’s post on InstructGPT and Alignment](https://openai.com/research/instruction-following)
- [Anthropic AI Alignment Papers](https://www.anthropic.com/index?subjects=alignment)
