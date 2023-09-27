# LLM-application-production
Source code and notes are based on EDX course Large Language Models: Application through Production

## LLMs are more than hype
They are revolutionize every industry.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/d9e24bd4-bfa9-4d7a-9e80-7669280e912b)

## LLMs are not that new
Why should we care about it now?
![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/b3f5fd46-8fd8-4758-8799-e9f153d57eff)

# NLP Ecosystem
![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/75873b5f-8def-41ab-9157-6ab4aef4f3ad)

At the top is Hugging Face, which we already mentioned. Its Transformers library, and the Hub associated with it, is probably best known for having a bunch of pre-trained deep learning-based NLP models and
pipelines. Hugging Face offers a lot more as well.
There are also a number of very popular libraries out there still doing classical NLP, not LLM or deep learning based.
There are some very famous proprietary ones like OpenAI, and then there are also some newcomers such as LangChain, which don't offer models per se but rather workflows or chains around models.

# Hugging Face
Hugging Face is a company and also a community of open source ML projects, most famous for NLP. Its Hub hosts models, datasets, and spaces for demos and code. These are easy to
download, under different licenses.
There are also a number of libraries it provides. The datasets library provides ways to download data from the Hub.
Transformers makes it easy to work with pipelines, tokenizers, models---these core components of NLP---and to download pre-trained ones from the Hub.

Let's take a look at Transformers first. What does the pipeline look like here?

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/ad9bea67-0d38-4ac1-acd2-26b1ac313001)

We're back to our summarization problem: article on the left, summary on the right, and the LLM in the middle.
Using Transformers, you can simply import the pipeline class, say I want a default summarization pipeline, and throw my article at it, and I'm done.

Under the hood it's picking a default LLM for me, configuring it, and trying to do the right thing.
In general though, we'd want to configure it further ourselves, so let's open up the pipeline. Here are common components it might have.

## Hugging Face Transformers Pipeline
![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/4716180e-3ef0-4efe-acc4-10c94b8faebe)

That raw input might go through something called prompt construction. We'll talk more about prompts a little later in the module. For now, just know that some LLMs---not all---require further instructions beyond the raw user input.
For summarization, it might be as simple as pre-pending summarize colon to that article.

That'll then go through a tokenizer which will encode the text as numbers, which is what our model the LLM will expect. The model outputs an encoded summary, and the tokenizer (the same tokenizer) decodes that summary.
I'll also note that this slide glosses over potential pre- and post-processing which might happen in the pipeline, but it gives the key components.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/77f69dcb-c1ca-49c0-9bdb-eaf8fff1f2da)

Looking at the tokenizer a little bit more closely, we can see that in the bottom left, it outputs that encoded data as input_ids, which is the actual encoded text, and then this attention_mask. We're not going to talk much about attention in this course. Suffice to say for now, **it is metadata describing the text which is output as input_ids**, and you need to pass it along to the model, which expects this
metadata.

On the right you can see us using this AutoTokenizer class. This is one of multiple Auto classes that Transformers offers, which sort of does the right thing when you give it the name of a model or tokenizer that
you want to load.
Given that tokenizer, we can pass in the articles, and then here's an example of some configurations you might specify:
max_length, whether or not to pad, whether or not to truncate the input. These are all around forcing variable length input text into fixed-length tensors which the LLM will expect.

You're going to adjust these based on the model you pick and also the task.

At the bottom, return_tensors is just saying we're using PyTorch.

Next comes the model.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/ca7175c5-fc20-4849-b5a3-c176e1728545)

We're going to pass in the input_ids and attention_mask and get encoded output.
On the right you can see us using another one of these Auto classes, this time for sequence-to-sequence language models.
We won't go into a ton of detail about the different classes of language models here, but this is essentially saying a variable length sequence of text (like our article) being transformed into a variable length
sequence of text (like our summary). Give it the model name. It loads the right model for us, and then we can pass in input_ids, attention_mask.

Here I'll note a little more detail about the metadata. This is where, you recall with the tokenizer we were specifying some parameters to handle variable length inputs. This is the metadata that will help the model handle those variable lengths.

Next come a few inference and output parameters: num_beams, which we'll explore more when we get to coding, is saying I want to do beam search to generate the output text. There are a number of ways to do inference, and this is one of them.
And then min and max length are saying:
I want between 5 and 40 tokens in my output summary. This is going to be adjusted of course to your requirements for the task.

That's enough about Transformers, so let's say a word about datasets. 

## Dataset
The library offers one-line APIs for loading and sharing datasets. We'll talk about NLP, but there's also audio and vision.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/e9628883-ec3c-424f-86c7-57d23840bc62)

The load_dataset API not only lets you specify the dataset name, but also a version number. That can be valuable for maintainability of code.
These datasets are hosted on Hugging Face Hub, and that lets you in the UI filter by things like task, size, license, language, and so forth, and also find related models which can be very useful.

# Model Selection
Let's talk about how to select the right model for a task.
Looking back back at our summarization application, I'll first note that you know we talk about these broad tasks: summarization, translation, whatever, but there are details.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/d5b44056-127e-483a-a805-3c2d1dbceaff)

For example with summarization something you may encounter later on when we get to coding is: do you want to do extractive summarization where you select representative pieces of text from the original, or
abstractive where you're generating new text?

Beyond that, once you figure out the details, how do you go about searching for it? You know, if you look at Hugging Face Hub, when I last looked it had 176,000 something models. I filtered by summarization. That gave me about a thousand models. And then what should I do?

Maybe sort by popularity? Maybe, but let's consider needs first. There are a lot of potential requirements and needs, and a lot of techniques to filter by them. Let's look at some easy choices first.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/983efe4a-d74a-4cfd-a32d-4e609eca1211)

Filtering by task in the upper left, license, language: these kinds of hard constraints can be pretty easy and pretty useful.
For example if you need a commercially permissive license, that can be pretty clear-cut.

In the bottom left, this isn't filtering within the UI which currently doesn't support it, but you can look up how large these models are, either in Files and Versions to get a ballpark idea of maybe
the number of gigabytes the PyTorch representation takes, or if the model is well documented the number of parameters it has.

That can be important if you need to limit hardware requirements for cost or latency or whatever.

In the upper right, you can see in the UI you can sort by popularity and updates. Popularity matters of course because people vote with their feet on what are good models.

Updates matter; especially if you are looking at a very old model, it might not even load properly with the latest Transformers library. If you want more details about what is in these updates to models, it can be useful to check the git release history. Well-documented models will document that.

Now getting into some more ambiguous ways of figuring out the right model---but still important ways---I want to talk about model variants, examples, and data.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/fb57481f-04ec-41e5-8851-7c1f51206e04)

On the left I'm recommending to pick good variants of models for your tasks. Here's what I mean. When a famous model, say T5 as shown in the image here, is published, it's often published with different sizes: base, a smaller version, a larger version, maybe an even larger version.

Here I'd say start prototyping with the smallest one, just to get moving quickly, keep costs low. You can always move to a bigger version which will presumably be more powerful.

Also look for fine-tuned variants of base models. So we'll talk more about fine-tuning later in the course, but basically if a model you pick has been fine-tuned on a dataset or a task very similar to yours,
it may perform better.

In the upper right, I'm recommending that you search not just for models but also for examples and datasets.

You know, this can be for many reasons, but I would say one of the top ones is that honestly not all models are well-documented, so a good example of usage will tell you not only what parameters you may want to tweak or whatever.
It can also help you avoid needing to know about model architectures. You don't need to be an LLM expert in order to pick a good model, especially if you can find where someone has already shown it to be good for your task.

Next ask: **is the model a generalist (good at everything) or was it fine-tuned to be great at a specific task**? Relatedly, which datasets were used for pre-training and/or fine-tuning?

**Fine-tuned models in general are going to be smaller and or perform better if they match the task that you are doing**. Ultimately though, it's about your data and users, so define KPIs and metrics, test on your data and users, and we're going to talk more about evaluation later in the course.

So those are some general tips on selecting a model. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/3252db9a-81a1-49ce-9c82-64521c369724)

The other part of selecting a model, I think, is recognizing famous good ones. Here is a short list, a small subset of the many models or model families out there. In the upper-right, there's a larger table of LLMs which you also might want to check out.

I'm not going to talk through this whole list, but I want to point out a few things. First a lot of these quote models are actually more model families, maybe ranging from millions of parameters to billions of parameters. And also a lot of them have been fine-tuned to be more specific.

So for example, I would call the first row Pythia sort of a family of base models, not necessarily meant to be used out-of-the-box (they could be), but perhaps more commonly used as base models further fine-tuned into, for example, the second line **Dolly which is a fine-tuned version of one of the Pythia models** for instruction following.

I'll also note that these models vary a lot in size, but you'll notice that some famous ones like maybe GPT 3.5 aren't actually as large as some of the less famous larger ones. So size does matter with LLMs---larger models tend to be more powerful, maybe be able to handle more types of tasks, but it's not everything. So for example with the GPT family, a lot of effort was put into techniques like the famous **Reinforcement Learning with Human Feedback**--- these alignment techniques to sort of give users what they're really looking for.

Other things which are really important of course are model architecture, what datasets were used for pre-training and/or fine-tuning, and these can cause major differences between these models.
That said, a lot of these foundation models really are interrelated, sharing or selecting from sort of a shared family of techniques or pre-training datasets.

# Common NLP Tasks
Let's take a look at some common NLP tasks.

Here's a short list. We're not going to talk about all of them in this module. We'll talk about the ones in bold. Also note that some of these quote tasks are very general and overlap with others.
But these terms are important to know because you'll see them in literature, on the Hugging Face Hub, and so forth.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/f2c5178c-abff-4f90-997d-f68c8332c94a)

A good example is the bottommost one: text generation. This can subsume almost any other task. Some of the summarization models we'll use actually are labeled as text generation because they do a bunch of
things including text generation.

## Sentiment Analysis
We already talked about summarization, so let's talk about sentiment analysis.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/ad7072da-7d98-43e0-a18b-3379fe1ad6df)

An example application using this might be, say, I monitor the stock market and I want to use Twitter commentary as an early indicator of trends. Given a tweet, is it positive, is it negative, maybe neutral, about whatever stock it's discussing?

You can see in the bottom-left that in addition to a label like positive or negative, you might want maybe a confidence score, and LLMs can often return that.

## Translation
The next task I'll mention is translation.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/223f6c33-343b-464e-8fab-b6ae4149c27c)

Here it's what it sounds like, but there are a couple things which are useful to point out.

Some LLMs are fine-tuned to go from one language to another specific language. For example the model in the upper-left is going from English to Spanish, and that's it.

Also note that sometimes you can specify the NLP quote task as a more general one like text-to-text generation. That's what we're doing with this model here.

In the bottom-left, the t5_translator model is a fairly general model, and there since it is general and can do multiple things beyond translation, we actually give it an instruction:
translate English to Romanian, and then give the user input.

## Zero-shot Classification
The next task we'll look at is zero-shot classification.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/3131ce1a-0477-4348-bb8b-4888869e6e4a)

A good example of this might be, say, I have a news browser, and given news articles I want to categorize sports, breaking news, politics, and so on.

Now I don't want to retrain a new model every time I change my categories. I want to use an existing model. And LLMs actually let you do this.

**Unlike classical machine learning, an LLM sort of already knows the language, and so when you add or change labels, it already knows what those labels mean. So it can potentially do that classification without being retrained**.

At the bottom-left, you can see how a pipeline would do this. Given the article and candidate labels, it'll return you its prediction. Really useful task.

## Few-shot Learning
The next one, few-shot learning, is very general and I'd almost call it a technique rather than a task.
In this example, you show the model what you want through examples, so **instead of fine-tuning a model for a specific task, we provide a few examples of doing that tas**k.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/8008cc59-a879-491b-8979-307620441219)

An LLM which is powerful enough can often sort of learn on the fly what you want to do. So on the right, you can see we're doing sentiment analysis, but using a model which is not designed for sentiment analysis.

Our instruction is: for each tweet, determine the sentiment. We give some examples: here's a tweet, here's a sentiment, a couple more. And then a query which is a new tweet, give me the sentiment.

This is a technique which you'd use when there is no fine-tuned model available for your task and you don't have enough labeled training data to train or fine-tune one, but you can write out a few examples.
This uses text generation models which must be quite general to understand and follow these instructions, but it's a really powerful technique.

We're starting to see here where the user input, which might be that query at the very bottom, is being crafted into a larger prompt.

# Prompts
Let's talk about prompts, which are our entry to interacting with powerful LLMs.

Prompts often appear in the context of what are called instruction-following LLMs.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/d59dc299-98a8-4a2b-9760-52409387f880)

To explain what these are, let's compare them with foundation models on the left. Foundation models are pre-trained on these very general text generation tasks, like given the text in blue, predict the next token in the sequence, and the next, and the next. Or fill in missing tokens in a sequence.

Instruction-following models on the other hand are tuned to follow almost arbitrary instructions or prompts. This is still very general but somehow a bit more specific.

Some examples might be: 
give me three ideas for cookie flavors; the LLM returns a numbered list.

Write a short story about some stuff; and it returns a short story.

Now, these are toy examples of course, but prompt engineering actually gets pretty serious, and we've seen some examples already.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/5ca5636c-fa8b-4a11-9992-222dcf1ae57f)

In general, prompts can be thought of as inputs or queries to LLMs to elicit responses---emphasis on elicit because you really are trying to tease out the right behavior from these otherwise black-box models.

We saw an example with our summarization problem. The T5 model expects a prefix summarize colon appended to that article, and we saw this in the code. That's all it expects, but it's enough to tell the model what you want it to do.

Now more generally, these prompts can be natural language sentences or questions, code, combinations of the above, emojis, pretty much any text.
They can also include outputs from other LLM queries. That's very powerful because it allows nesting or chaining LLMs, making these complex and dynamic interactions.

We saw a more complicated example with few-shot learning where a prompt had an instruction, a number of examples to sort of teach the LLM what we wanted, and then the actual query.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/a5e67044-ecd7-4e86-9a27-97b3a2eac5e5)

It gets even more so. LangChain generated this structured output extraction example for me, where it has a number of parts. At the top a very high-level instruction: answer the user query; it should fit this
format. An explanation and example of how to understand the desired output format, specification of that output schema, and then the final instruction: tell me a joke.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/9c9d8b4d-ad10-4073-a5c7-b924fc9d43f8)

Now this prompt is not a joke even though it looks ridiculous. This actually works with some models, and it outputs a structured format which can then be fed into downstream data pipelines.
This is just a nice example of how powerful these prompts and prompt engineering can be.

# Prompt Engineering
Let's talk about prompt engineering. We're going to have a lot of general tips, but we're going to start off with a caveat that **prompt engineering is model-specific**.

So prompts will guide a model to complete the task in the way you wanted, but different models may require different prompts. And a lot of guidelines you'll see out there are specific to one of the most popular services, **ChatGPT and its underlying OpenAI models**. They may not work for non-ChatGPT models, but a lot of the techniques do carry over, even if the specific texts of the prompts do not.

Different use cases may require different prompts, and so iterative development is key, hence engineering.

## General Tips
Let's start with some general tips around how a good prompt needs to be clear and specific. Just like when you ask a human to do something you need to be clear and specific, that helps with LLMs as well.

A good prompt often consists of an:
- instruction,
- some context or background information,
- an input or question,
- output type or format.

You should describe the high level task with clear commands. That may mean:
- Use specific keywords like Classify, Translate, so on,
- Or including detailed instructions.

And finally, this is engineering, so test different variations of the prompt across different samples. Use a data-driven approach here: what prompt does better on average for your set of inputs?

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/3641c7c2-e883-4400-8506-e4c801c708e8)

Just a refresher: we saw this example in the last video of LangChain giving a prompt with different components: a very clear instruction, context or an example, specification of the output format, and
then the actual user query input: tell me a joke.

## How to help the model to reach better answer
There are also techniques for helping the model to reach a better answer, to sort of think better.
- First, you can ask the model not to make things up. You've probably heard of the term hallucination, where models will sometimes just spout nonsense or false things. But you can tell the model not to, and that can help.
- You can also ask the model not to assume or probe for sensitive information, and finally this last one is very powerful.
- Ask the model not to rush to a solution, but instead take more time to think using what's called chain of thought reasoning. Things like: explain how you would solve this problem, or do this step-by-step. That often leads to better results.

Prompt formatting can also be important.

## Prompt Formatting Tips
Use delimiters to distinguish between the instruction and the context. Also use them to distinguish between the user input, if this is a user-facing application, and the prompt that you add around it.
Ask the model to return structured output, and provide a correct example.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/b78df92d-7635-4ac9-bcd1-45cd8cb5eee3)

On the right you can see where a user-facing application allows users to input part of a prompt but then we wrap it with a larger prompt. And in this example if you can read it, the user is trying to override our
instruction. This is called **prompt injection**, and prompt formatting can help avoid this.

This starts to get into the idea of hacking prompts, which is exploiting LLM vulnerabilities by manipulating inputs. But good prompts can help reduce this.

## Good prompts reduce successful hacking attempts

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/9c5cce56-46a3-4f5e-b59a-c84d721e4b08)

On the upper-left, here's prompt injection which we just saw, where you're basically trying to get the LLM to ignore the real instruction which the application wants it to follow, and instead override it with a user input instruction.
In the bottom-left is prompt leaking, where we're extracting sensitive information from a model. Here is a public example of extracting the secret code name of Microsoft Bing Search. On the right is an example of jailbreaking, where you're bipassing a moderation rule.

Here it's asking how to do something illegal, and the model first says, I can't tell you. And then some rephrasing, and the model actually answers the user question. This, like any computer security thing, is sort of a constant battle between the people developing these applications and the people trying to break them. And so this example on the right actually doesn't work anymore because it has been fixed. This is something you might need to think about in your applications as well.

## How else to reduce prompt hacking

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/da95c735-43b0-44b8-8cbc-61c9aa36af25)

There are other techniques for reducing prompt hacking.
-  You can post-process or filter.
  - Use another model to clean the output, or tell the model to remove all offensive words from the output.
- You can repeat instructions or sandwich instructions at the end. This can help llm pay attention to whatyou really wanted to do.
- You can enclose user input with random strings or tags. That makes it easier for the model to distinguish what the user input is versus your instructions.
- And if all else fails, it can help to select a different model or restrict prompt length.

# Resources
Here are some guides and tools to help with writing prompts. Some are OpenAI-specific, and some are not, but these will be great resources as you dive into the lab.
[Best practices for OpenAI-specific models](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api), e.g., GPT-3 and Codex
[Prompt engineering guide](https://www.promptingguide.ai/) by DAIR.AI
[ChatGPT Prompt Engineering Course](https://learn.deeplearning.ai/chatgpt-prompt-eng) by OpenAI and DeepLearning.AI
[Intro to Prompt Engineering Course](https://learnprompting.org/docs/intro) by Learn Prompting
[Tips for Working with LLMs](https://github.com/brexhq/prompt-engineering) by Brex

Tools to help generate starter prompts:
• [AI Prompt Generator](https://coefficient.io/ai-prompt-generator) by coefficient.io
• [PromptExtend](https://www.promptextend.com/)
• [PromptParrot](https://replicate.com/kyrick/prompt-parrot) by Replicate

