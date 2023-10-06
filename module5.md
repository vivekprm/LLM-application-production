# Society and LLMs
Largest risks with LLMs to increase awareness of them and make sure people understand the limitations and also some ways available to mitigate some of these. So, LLMs can produce, you know, various kinds of unreliable behavior. So I'll just mention two of them, you know, really quickly and you know we'll talk about what to do about them.
- One of the first, you know, sources of bad behavior has to do with the language modeling itself. Right, if you train your LLM on a whole bunch of data, especially if from the web, the data might have biases in some fashion that you don't want your application to have you know. Maybe so that's bad because it learns the bias, because it just learns the language that's out there.
- The data might also be wrong about certain things, you know, you don't want a language model that's asked a medical question and returns the popular wisdom from the web, when you know you're building an application, where, like you know, it has to be something approved, by you know, by the government or it has to be the latest scientific knowledge or something like that.

There's just a lot of junk on the web, you know, there's offensive text, there's like search engine optimization, so it's not necessarily great. So that's like one problem, your language modeling and you're just getting, you know, you're modeling the wrong facts that you don't want to learn.

- Second example is what's called hallucination, where the model when it doesn't really know, you know, what to say, it will still make something up and it can sound
very confident about it. 

## Risks and Limitations
Even though LLMs have the great capability to apply to many different industries and transform them, they also come with their own risks and limitations.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/26fa8955-bbd9-4e4c-9b91-2b21fd5a26ff)

Many of these risks and limitations are really hard to mitigate, especially if there are intentional and irresponsible, and malicious actors. But we'll first look at the source that enables LLM power today, which is the data, and look at how data can also translate to model bias.

While it's great that LLMs can improve content creation efficiency, **we also have to acknowledge that it undermines the creative economy**. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/6ed61e30-6e53-451c-ab4d-5ad6aa0bb2d5)

You probably have heard about the debate: you know, **is AI-generated art still art**? We are not here to answer that question. But it does shed some insight into the uncertainty around how much we can trust the content out there. And are we actually jeopardizing the artist industry by not giving them attribution or maybe even infringing upon their copyrights?

# Automation displaces job and increases inequality
Automation can also displace jobs. The U.S Bureau of Labor Statistics shows that the number of customer service employees will decline by four percent in 2029. And a lot of new roles introduced by AI may also have limited career opportunities in terms of skill development and wage gain margin. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/ffecaa69-75f4-4f17-a26b-b8e6e4ef0a04)

For example, if someone were to work as a data labeler, which means that you are doing the manual work of classifying text into categories or maybe label this text as toxic or not, so this type of jobs are likely not having a lot of growth for a particular person in terms of a career progression. But there are also reports to show that human annotators for toxic text classification often have a higher chance of exhibiting poorer mental health because of the regular and constant exposure to toxic content.

And for many of us who are watching this video, who have internet access or who have access to cutting-edge technology, it means that these technology advancements can serve us a lot of good. But it also does mean that less privileged people in less developed countries will not be able to take advantage of the technology; hence, we are increasing the global inequality.

# Incurs environmental and financial cost
LLMs also incur a lot of environmental and financial costs. On average, a typical U.S person will emit three times as much carbon dioxide than a global average person. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/df35a635-495b-4e6a-a1cd-012f2e4353b6)

Training a large language model from scratch is very costly. So here is an estimate published by a paper: for every thousand parameters, that roughly costs you one dollar to train the model. And for a Chat-GPT model to be trained from scratch, you probably have taken **175 million dollars**. So this high cost means that most organizations either have only one shot to do it right or maybe even no shot at all, especially when you are really a small business or maybe when you are an individual.

# Big training data does not imply good data
And in today's age, we are really fortunate that we can build powerful models from big data. However, we must remember that big data doesn't always imply good data. Remember that most data that go into model training comes from the internet. And if your grandparents are like mine, then they probably don't use the internet all that much. It means that **internet data tends to over-represent younger people** and also those from developed countries too.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/3073f7fa-8bc3-4042-957a-39ce6c4a1471)

It goes without saying that most of the training text are geared towards English and perhaps not as unsurprisingly, specifically in the UK and also in the US.

This paper from 2021 also points out that GPT-2 data is sourced from Reddit outbound links. But almost 70 percent of Reddit users are men and over 60 of the users are younger than 29 years old.

Similarly, only up to 15 of Wikipedia entries are about females. So the idea behind this paper is that perhaps the language models that we're using today are really not that smart; they are simply parrots, who are really good at copying what the humans say. It means that if the data is bad, we probably cannot imagine these language models to do much better.

So we talked about how the size of the data doesn't guarantee the diverse representation of data. But what is also really challenging to do is to audit the data. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/1443e7a1-67a5-4fdd-b87e-d7f32a5f7a64)

**Do we actually have good data quality? But how do we even start auditing when the data is so big?**
So when the mode, the data input is biased, we can almost certainly expect the model to be biased as well. "Garbage in, garbage out" is an old adage that still applies to language models.

But the other fundamental limitation with the data today is that **only certain types of stories make into the news**. For instance, **a peaceful protest is much less eye-catching on the newspaper than a violent protest**. Therefore, the former often goes unreported, which means our model doesn't know about it. But the other limitation is that we cannot afford to update our model all that much, even if we can update our data. Because we established earlier that training a model from scratch is expensive so when we cannot update our data then we risk having a outdated model.

# Models can be toxic, discriminatory, exclusive
Models can be highly toxic, discriminatory, and also exclusive. And the reason is because our data is flawed. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/0b5997cc-06b3-407b-88eb-2c348ac5cfee)

So if you look at the examples on this slide over here, we see that, on the right hand side, we have many more females represented in a family context versus a politics context. So in fact, the paper actually found that female-sounding names are depicted often as less powerful. And we can argue that this is a reflection of the society,
but it does also mean that we need to carefully consider how to use such a model when it can embed bias that we may not necessarily want the model to incorporate.

There are also some other models that exhibit certain bias against certain demographic groups. And it's also not surprising that these models can have poorer performance for some languages as well because of the data problem that we mentioned.

# (Mis)information hazard
Compromise privacy, spread false information, lead unethical behaviors

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/30ad75a7-bf27-463f-b437-784432e32839)

The next risk has to do with information hazards. So this comes in two prongs. The first is when we can accidentally compromise privacy by leaking or inferring private information.

So this slide over here shows how **Sydney, the Microsoft chatbot**, accidentally reveals itself to be Sydney and employees can also accidentally leave company secrets by interacting with another close-sourced model. What is really interesting but also concerning is the image on the bottom over here, where it shows that the LLM can confidently output information that is incorrect.

In fact, it suggests that violence within the couple can actually be good. 

# Malicious uses
LLMs can also facilitate many, many malicious use cases. For example, fraud, censorship, surveillance, or cyber attacks. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/77f7e526-ad65-4eb2-9ed3-a5496b547f71)

# Human-computer interaction harms
And lastly, this is something that we're all prone towards as well, which is when we are relying on this technology way too much and when we give way too much trust to these models. For example, if I were to struggle with mental health it will probably not be wise for me to consult a chatbot on what to do.

Many of this generated text indicate that large language models tend to hallucinate.

# Hallucination
From this paper in 2022, it indicates that hallucination refers to when the generated content is nonsensical or unfaithful to the provided source content. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/f37e1d40-cfbf-4fb4-a642-93367d069494)

It means that the output can sound completely natural and fluent and it also means that the output can sound really confident even when it's wrong.

## Intrinsic vs. extrinsic hallucination
There are two types of hallucination: intrinsic and extrinsic. Based on individuals, we may all have different tolerance levels based on how faithful or how factual we expect these outputs to be. And we'll talk in just a second about what faithful actually means in this context.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/abd6a5c6-e66d-4fe9-a0de-bcd8d43623a9)

**Intrinsic refers to when the output directly contradicts the source**. If I give a source text that indicates the first Ebola vaccine was first approved, was approved
by the FDA in 2019. But a summary output indicates that the first Ebola vaccine was approved in 2021.
So this is a very clear case of contradiction and it means that the output is not faithful to the source text over here. And it also means that this output is completely not factual as well.

On the other hand, for **extrinsic hallucination**, it refers to when we cannot verify the output from the source, but the model itself might not be wrong.
For example, if I were to say Alice won first price in fencing last week and then the model tells me that Alice won first price in fencing for the first time last week and she was ecstatic.
It's probably true that Alice did that for the first time and she was really excited that she got it, but we cannot verify that from the source. It means that we cannot really say that output is factual or faithful towards the source.

## Data leads to hallucination
So what leads to hallucination? The first component, probably without any surprises, is the data. **So how we collect data matters a lot**, in terms of how the model performs. And we talked about in the earlier segment that when we have big data, it's really hard to do audits well or do any audits at all. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/a3c9974e-ba95-403f-b4ef-52d79e26edce)

Same goes to the data collection process. We may gather any text that we have without any factual verification. We also do not filter out exact duplicates most of the
time. For example, if you were to ingest the same Reddit thread twice, that counts as a duplicate and duplicates can bias the model. If we have many of the same Reddit threads show up in the data, then it means that it's more likely for the model to output responses from those Reddit threads.

But the other problem regarding the data is actually just regarding how **open-ended these generative tasks are**. For example, in a chat application, we will probably want the chatbot to be more engaging. And therefore, it means that we would expect more diverse responses. If I were to ask the chatbot about the same thing many times, and if the chatbot will always repeat the same things, it will probably be a chatbot that we won't use for very long. 

So we want the chatbot or some applications to have more diversity to improve engagement, but this type of diversity can also correlate with bad hallucination, especially when we need factual and reliable output. When we ask the chatbot about medical literature, our tolerance level for anything that is non-factual will be quite low, compared to when we ask something about how to make a perfect salad.

But this open-ended nature of generative task is just a really hard-to-avoid problem and it's something that will have to deal with as a user of LM applications.

## Model leads to hallucination
The second prong that leads to hallucination is the model. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/6ee7180d-6c1a-412c-977f-3d274e0093bc)

- The first reason within the model itself is **imperfect encoder learning**. It means that encoder learns wrong correlations between parts of training data.
- The second reason can happen at decoding time, which means that when the model is trying to generate text output, the decoder actually attends the wrong part of the input source, But there are also decoder design, that's by design, encouraging randomness and also unexpectedness. For example, top-K sampling. So for those types of decoders, rather than picking the most likely token, it would select any one of, out of the four candidates over here, that you see on the slide, to generate the next token.
- The third reason is exposure bias. So very technically speaking, this means that there is a discrepancy in decoding between the training and also inference time. But plainly speaking, it means that model tends to generate output based on its own historically generated token. So it also means that the model can veer off a topic. When you start off asking about dishwasher, maybe the model itself would then start generating content about the dryer.
- The fourth reason has to do with parametric knowledge bias. As summary, it means that the model will stick to what it knows. So all models tend to generate output based on what it has memorized, rather than the provided input.

## Evaluating hallucination is tricky and imperfect
Evaluating hallucination is tricky and imperfect, as I mentioned before, different individuals can have different expectations about how the models actually behave and we can also have very different decision criteria to determine whether a certain content is toxic or why does certain content is classified as misinformation. There are two categories of metrics here that we can rely on to assist with evaluating hallucination. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/dcac89a7-840a-4a96-910f-cd790ea43b05)

### Statistical Metrics
The first category is **statistical metrics**: BLEU, ROUGE, and METEOR have been around for some time in NLP and when using these metrics, we see that approximately 25 of summaries contain hallucination, which means it contains unsupported, very unsupported information.

The second metric over here is called **PARENT**, which measures the hallucination using both source and also output text. It means that it's actually using n-grams behind the scene to capture what is in the source versus the target and then it calculates the **F1 score**.

The third type of metric is called **BVSS** which stands for **Bag-of-Vectors Sentence Similarity**. It measures whether the translation output has the same amount of information as the translation reference.

### Model-based Metrics
The second category is **model-based metrics**. It means that we are leveraging another model to help us evaluate hallucination. But this category of metrics also means that any of the errors from these models that we're leveraging also get propagated throughout as well.
So the first type of model that we can leverage is **information extraction**. And this is especially useful for any named entity recognition use cases. We are trying to extract knowledge so we can use this to compare with a language-based, the language model output.

The second metric over here is **question-answering-based**. It means that we can measure the faithfulness by measuring the similarity among the same, among the different answers to the same question.

The third metric over here is **faithfulness**. It asks the question "does the output actually contain any unsupported information?"

The last one over here is **language-model-based**, which means that we are using a language model to help us calculate the ratio of hallucinating tokens to the total number of target token. So as you can see over here, there are a variety of metrics to help us evaluate hallucination, but none of them is perfect. 
