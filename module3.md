# Multi-stage Reasoning
This is a really exciting capability that people are developing with LLMs basically building more complicated applications out of individual calls to language models. So in this module we're going to start off by talking more about prompting and how to how to build more sophisticated prompts that get a model to reason in some way to to produce better answers. 

One of the common tools that we'll talk about this is prompt templates and there are various techniques like Chain of Thought that can guide the model into producing the
kind of output you want using you know its statistical capabilities. And next we'll move to the concept of chaining prompts which allows you to create you know steps in your LLM application that are good at a particular subtask and put them together into something bigger that you know overall works well for a complicated task. 

So we'll talk about how this helps you know break down the problem make it easier than you know than just having a single giant prompt and hoping it gets everything right. 
We'll talk about open source frameworks in this space like **LangChain** and basically this idea of chaining and of modularizing an application you know much like in software engineering is allowing people to build applications using language models that that can do sophisticated things and are still maintainable and controllable and high quality. 

We'll also touch a little bit on the area of **LLM Agents** so this is a llm application that can use external tools you know much in the way that a model can generate text as an answer it could generate text that's actually you know a call an API call to some kind of tool and then you can sort of you know get a model to use these tools to do you know things that it couldn't do using using knowledge alone, so that also fits very well into this chaining paradigm. 

And we'll also talk about sort of autonomous LLM systems that kind of discover which tools to use and how to do things as though. This area of multi-stage reasoning is also near and dear to my heart as a researcher. We actually have a project at Stanford called demonstrate search and predict or DSP which is looking at a way to build
pipelines of multiple LLM calls that are highly reliable and can also automatically be improved using data to make them better. And again this is one of the fastest moving areas in LLMs but also really powerful really important to get your LLM to plug into other systems and do you know cool things with other software.
