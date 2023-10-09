# LLMOps
So much like in machine learning we have this field of MLOps of how to maintain and operate applications, we need the same for large language models and a lot of you know classical Ops ideas apply but large language models are also different in some ways. You know they generate this unstructured text, they have different things that affect the, uh so it makes sense to sort of rethink that and look at what goes into that. so we'll talk about some of today's best practices for LLM Ops for the whole
stack, not just the LLM but also the vector databases the chains end-to-end applications everything from monitoring, to improving quality, doing development in a in a flexible and collaborative way testing, and getting high performance in your application so you can build a rock solid production LLM application.

# Learning Objectives
By the end of this module you will:
- Discuss how traditional MLOps can be adapted for LLMs.
- Review end-to-end workflows and architectures.
- Assess key concerns for LLMOps such as cost/performance tradeoffs, deployment options, monitoring and feedback.
- Walk through the development-to-production workflow for deploying a scalable LLM-powered data pipeline.

# MLOps
ML and AI are becoming critical for businesses

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/8eacbf59-75ac-4e4e-8bc0-3897b56de3c3)

Goals of MLOps
- Maintain stable performance
  - Meet KPIs
  - Update models and systems as needed
  - Reduce risk of system failures
- Maintain long-term efficiency
  - Automate manual work as needed
  - Reduce iteration cycles dev→prod
  - Reduce risk of noncompliance with requirements and regulations

# Traditional MLOps
I like to define MLOps as DevOps + DataOps + ModelOps. 

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/8e782753-0404-470a-b2db-092135694acf)

That means a set of processes and automation for managing ML assets like code, data, and models to improve performance and long-term efficiency, our two goals from the last video.

When you think of MLOps, you may think of things like a dev-staging-prod workflow, testing and monitoring, CI/CD. We won't have time to delve into all these details in this course but for those who want more background, look at [this eBook](https://www.databricks.com/resources/ebook/the-big-book-of-mlops) for more details.

# Traditional MLOps architecture
Instead, I want to talk through a high level reference architecture for traditional MLOps shown here.

![image](https://github.com/vivekprm/LLM-application-production/assets/2403660/c35c0b67-7195-4c9e-96dc-699d6299baeb)

Like all good reference architectures, this is a generalization, but I think it highlights a lot of key ideas. At the top, you'll see source control for managing code. At the bottom, a Lakehouse data layer, and going left to right in the middle are development, staging, and production environments.

Starting on the left, a data scientist for example might operate in the development environment. Maybe doing some exploratory data analysis--- that code might not move towards production---but also writing pipelines like a model training pipeline, a feature table refresh pipeline, and these are geared towards production.

Once those are ready, they can be committed to source control, and that is going to be one of our primary conduits for moving ML assets---code in particular---towards production.
At the bottom is our Lakehouse data layer.

We won't have time to cover what a Lakehouse is, but I think key elements here are that it is a shared data layer across a possibly diverse set of tools forming this broader MLOps set of environments and systems.

Having shared access, but with access controls, is very important, for a Data Scientist for example operating in the development environment might need read access to production data in order to debug something out there, but they certainly shouldn't have write access. And so this kind of flexible control with shared access to this data and a single source of truth is really valuable.

Now, as our code moves towards the staging environment, it goes through Continuous Integration or CI tests. That basically breaks down into quick unit tests (Does this piece of code work in isolation?) and longer integration tests (Does this piece of code work alongside all the other pipelines and services it'll see in production?).

That last point is crucial: the staging environment needs to mimic the production environment as closely as reasonable. That means the same set of services, the same set of pipelines, and so in this simplified reference diagram, we're not showing all those pipelines explicitly, but when we look at production, remember that those
same pipelines and services are instantiated in staging, just maybe in a smaller setting (less data for faster tests).

Once tests have passed, that code can move towards production. Here we've zoomed in a bit, showing all those pipelines which were being developed in Dev, tested in Staging, and now are instantiated in Production.

Going from the bottom left, we have data being read into a feature table refresh job, maybe written out to a feature table. That could be done in a batch mode, streaming, whatever. That might be fed into, say, an automatic model retraining pipeline, maybe run once a week.

When a new model is produced, it is put into that model registry at the top layer. If you're not familiar with MLflow or model registries, think of them like opinionated repositories for models. Opinionated in the sense that they come with statements of: this model has this many versions, and for this version it is in a stage development, staging, or production, basically tracking these different model versions as they move towards production readiness.

Speaking of moving them towards production, that's what the Continuous Deployment or CD pipeline does. It puts those through either incremental rollout or stages of tests and eventually marks them ready for production, at which point they can be loaded into inference and serving systems on the right and also monitored.

# Module 6 Resources
**General MLOps**
- [“The Big Book of MLOps”](https://www.databricks.com/resources/ebook/the-big-book-of-mlops) (eBook overviewing MLOps)
  - Blog post (short) version: [“Architecting MLOps on the Lakehouse”](https://www.databricks.com/blog/2022/06/22/architecting-mlops-on-the-lakehouse.html)
  - MLOps in the context of Databricks documentation ([AWS](https://docs.databricks.com/machine-learning/mlops/mlops-workflow.html), [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/mlops/mlops-workflow), [GCP](https://docs.gcp.databricks.com/machine-learning/mlops/mlops-workflow.html))

**LLMOps**
- Blog post: Chip Huyen on [“Building LLM applications for production”](https://huyenchip.com/2023/04/11/llm-engineering.html)

**[MLflow](https://mlflow.org/)**
- [Documentation](https://mlflow.org/docs/latest/index.html)
  - [Quickstart](https://mlflow.org/docs/latest/quickstart.html)
  - [Tutorials and examples](https://mlflow.org/docs/latest/tutorials-and-examples/index.html)
  - Overview in Databricks ([AWS](https://docs.databricks.com/mlflow/index.html), [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow/), [GCP](https://docs.gcp.databricks.com/mlflow/index.html))

**[Apache Spark](https://spark.apache.org/)**
- [Documentation](https://spark.apache.org/docs/latest/index.html)
  - [Quickstart](https://spark.apache.org/docs/latest/quick-start.html)
- Overview in Databricks ([AWS](https://docs.databricks.com/spark/index.html), [Azure](https://learn.microsoft.com/en-us/azure/databricks/spark/), [GCP](https://docs.gcp.databricks.com/spark/index.html))
- [Setup Apache Spark with Delta Lake](https://docs.delta.io/latest/quick-start.html#set-up-apache-spark-with-delta-lake)

**[Delta Lake](https://delta.io/)**
- [Documentation](https://docs.delta.io/latest/index.html)
- Overview in Databricks ([AWS](https://docs.databricks.com/delta/index.html), [Azure](https://learn.microsoft.com/en-us/azure/databricks/delta/), [GCP](https://docs.gcp.databricks.com/delta/index.html))
- [Lakehouse Architecture (CIDR paper)](https://www.cidrdb.org/cidr2021/papers/cidr2021_paper17.pdf)
- [Deltalake docker](https://github.com/delta-io/delta-docs/tree/main/static/quickstart_docker)
