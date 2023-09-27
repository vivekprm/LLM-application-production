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


