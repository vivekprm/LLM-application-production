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
