**Question 1**  
**What problem can occur if there is not enough overlap between consecutive chunks when splitting a document for an LLM?**  
    ○ The embeddings of the consecutive chunks may be more similar semantically.
○ It will not increase the number of chunks of a given size.
○ It will not have any impact.
✅ **The continuity of the context may be lost.**

**Question 2**
**When using a specific LLM and splitting documents into chunks, which parameter should you check to ensure the chunks are appropriately sized for processing?**
    ✅ **Context window size.**
○ Max number of tokens LLM can generate.
○ Number of LLM parameters.
○ Number of LLM layers.

**Question 3**
**An AI development company is working on an advanced AI assistant capable of handling queries in a seamless manner.****
Their goal is to create an assistant that can ****analyze**** images provided by users and generate descriptive text, as well as take text descriptions and produce accurate visual representations.**
**Considering the capabilities, which type of model would the company likely focus on integrating into their AI assistant?**
    ○ A language model that operates on a token-by-token output basis.
✅ **A diffusion model that specializes in producing complex outputs.**
○ A Retrieval-Augmented Generation (RAG) model that uses text as input and output.
○ A Large Language Model-based agent that focuses on generating textual responses.


**Question 4**
**A student is using OCI Generative AI Embedding models to summarize long academic papers.****
If a paper exceeds the model’s token limit, but the most important insights are at the beginning, what action should the student take?**
    ○ Split the paper into multiple overlapping parts and embed separately.
✅ **Select to truncate the end.**
○ Manually remove words before processing with embeddings.
○ Select to truncate the start.

**Question 5**
**What is the primary function of the “temperature” parameter in OCI Generative AI Chat models?**
    ○ Determines the maximum number of tokens the model can generate per response.
○ Assigns a penalty to tokens that have already appeared in the preceding text.
✅ **Controls the randomness of the model’s output, affecting its creativity.**
○ Specifies a string that tells the model to stop generating more content.

**Question 6**
**What is the purpose of the given line of code?**
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)
    ✅ **It loads the OCI configuration details from a file to authenticate the client.**
○ It defines the profile that will be used to generate AI models.
○ It establishes a secure SSH connection to OCI services.
○ It initializes a connection to the OCI Generative AI service without using authentication.


**Question 7**
**What is the significance of the given line of code?**
chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
    model_id="ocid1.generativeaimodel.oc1.eu-frankfurt-1.amaaaaaask7dcey..."
)
    ○ It creates a new generative AI model instead of using an existing one.
○ It sets up the storage location where AI-generated responses will be saved.
✅ **It specifies the serving mode and assigns a specific generative AI model ID to be used for inference.**
○ It configures a load balancer to distribute AI inference requests efficiently.

**Question 8**
**In the given code, what does ****setting**** truncate = "NONE" do?**
embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
    model_id="cohere.embed-english-v3.0"
)
embed_text_detail.inputs = inputs
embed_text_detail.truncate = "NONE"
    ✅ **It prevents input text from being truncated before processing.**
○ It ensures that only a single word from the input is used for embedding.
○ It forces the model to limit the output text length.
○ It removes all white space from the input text.


**Question 9**
**What distinguishes the Cohere Embed v3 model from its predecessor in the OCI Generative AI service?**
    ○ Support for tokenizing longer sentences
○ Emphasis on syntactic clustering of word embeddings
○ Capacity to translate text in over 20 languages
✅ **Improved retrievals for Retrieval-Augmented Generation (RAG) systems**

**Question 10**
**Which of the following statements is NOT true?**
    ✅ **Embeddings are represented as single-dimensional numerical values that capture text meaning.**
○ Embeddings can be used to compare text based on semantic similarity.
○ Embeddings of sentences with similar meanings are positioned close to each other in vector space.
○ Embeddings can be created for words, sentences and entire documents.

**Question 11**
**In OCI Generative AI Agents, what happens if a session-enabled endpoint remains idle for the specified timeout period?**
○ The session remains active indefinitely until manually ended.
○ The session restarts and retains the previous context.
○ The agent deletes all data related to the session.
✅ **The session automatically ends and subsequent conversations do not retain the previous context.**

**Question 12**
**What happens when you enable the session option while creating an endpoint in Generative AI Agents?**
✅**The context of the chat session is retained, and the option cannot be changed later.****
**○ The agent stops responding after one hour of inactivity.
○ The context of the chat session is retained, but the option can be disabled later.
○ All conversations are saved permanently regardless of session settings.

**Question 13**
**Which option is available when moving an endpoint resource to a different compartment in Generative AI Agents?**
○ Create a duplicate endpoint in the new compartment manually.
○ Modify the endpoint's data source to match the new compartment.
✅ **Select a new compartment for the endpoint and move the resource.**
○ Archive the endpoint before moving it to a new compartment.

**Question 14**
**What happens when you delete a knowledge base in OCI Generative AI Agents?**
○ The knowledge base is archived for later recovery.
✅ **The knowledge base is permanently deleted, and the action cannot be undone.**
○ Only the metadata of the knowledge base is removed.
○ The knowledge base is marked inactive but remains stored in the system.

**Question 15**
**You have set up an Oracle Database 23c AI table so that Generative AI Agents can connect to it. You now need to set up a database function that can return vector search results from each query.****
What does the score field represent in the vector search results returned by the database function?**
○ The top_k rank of the document in the search results
○ The token count of the BODY content
✅ **The distance between the query vector and the BODY vector.**
○ The unique identifier for each document.

**Question 16**
**In OCI Generative AI Agents, if an ingestion job processes 20 files and 2 fail, what happens when the job is restarted?**
✅ **Only the 2 failed files that have been updated are ingested.**
○ All 20 files are re-ingested from the beginning.
○ The job processes all 20 files regardless of updates.
○ None of the files are processed during the restart.


**Question 17**
**How should you handle a data source in OCI Generative AI Agents if your data is not ready yet?**
✅ **Create an empty folder for the data source and populate it later.**
○ Leave the data source configuration incomplete until the data is ready.
○ Upload placeholder files larger than 100 MB as a temporary solution.
○ Use multiple buckets to store the incomplete data.

**Question 18**
**How are fine-tuned customer models stored to enable strong data privacy and security in OCI Generative AI service?**
✅ **Stored in OCI Object Storage and encrypted by default.****
**○ Stored in OCI Key Management service.
○ Stored in an unencrypted form in OCI Object Storage.
○ Shared among multiple customers for efficiency.

**Question 19**
**How long does the OCI Generative AI Agents service retain customer****‐****provided queries and retrieved context?**
✅ **Only during the user’s session.**
○ Indefinitely, for future analysis.
○ Until the customer deletes the data manually.
○ For up to 30 days after the session ends.

**Question 20**
**What does a dedicated RDMA cluster network do during model fine-tuning and inference?**
    ○ It increases GPU memory requirements for model deployment.
○ It leads to higher latency in model inference.
✅ **It enables the deployment of multiple fine-tuned models within a single cluster.**
○ It limits the number of fine-tuned models deployable on the same GPU cluster.


**Question 21**
**Which role does a “model endpoint” serve in the inference workflow of the OCI Generative AI service?**
    ○ Evaluates the performance metrics of the custom models.
○ Hosts the training data for fine-tuning custom models.
✅ **Serves as a designated point for user requests and model responses.**
○ Updates the weights of the base model during the fine-tuning process.

**Question 22**
**A startup is evaluating the cost implications of using the OCI Generative AI service for an app that generates text. They anticipate a steady but moderate volume of requests. Which pricing model would be most appropriate for them?**
    ○ Dedicated AI clusters, as they offer a fixed monthly rate regardless of usage.
○ On-demand inferencing, as it provides a flat fee for unlimited usage.
○ Dedicated AI clusters, as they are mandatory for any text-generation tasks.
✅ **On-demand inferencing, as it allows them to pay per character processed without long-term commitments.**

**Question 23**
**How do Dot Product and Cosine Distance differ in their application to comparing text embeddings in NLP?**
    ✅ **Dot Product measures the magnitude and direction of vectors, whereas Cosine Distance focuses on the orientation regardless of magnitude.**
○ Dot Product is used for semantic analysis, whereas Cosine Distance is used for syntactic comparisons.
○ Dot Product calculates the literal overlap of words, whereas Cosine Distance evaluates the stylistic similarity.
○ Dot Product assesses overall similarity in content, whereas Cosine Distance measures topical relevance.


**Question 24**
**How does Retrieval-Augmented Generation (RAG) differ from prompt engineering and fine-tuning in terms of setup complexity?**
     ○ RAG is simpler to implement as it does not require training costs.
✅ **RAG is more complex to set up and requires a compatible data source.**
○ RAG involves adding LLM optimization to the model’s prompt.
○ RAG requires fine-tuning on a smaller domain-specific dataset.

**Q****uestion 25**
**Which is a distinguishing feature of “Parameter-Efficient Fine-Tuning (PEFT)” as opposed to classic fine-tuning in Large Language Model training?**
    ✅ **PEFT involves only a few or new parameters and uses ****labeled****, task-specific data.**
○ PEFT modifies all parameters and uses unlabeled, task-agnostic data.
○ PEFT modifies all parameters and is typically used when no training data exists.
○ PEFT does not modify any parameters but uses soft prompting with unlabeled data.

**Question 26**
**How can you verify that an LLM-generated response is grounded in factual and relevant information?**
    ○ Examine the document chunks stored in the vector database.
✅ **Check the references to the documents provided in the response.**
○ Use model evaluators to assess the accuracy and relevance of responses.
○ Manually review past conversations to ensure consistency in responses.

**Question 27**
**Which category of pretrained foundational models is available for on-demand serving mode in the OCI Generative AI service?**
    ○ Generation Models
○ Summarization Models
✅ **Chat Models**
○ Translation Models

**Question 28**
**You are developing a chatbot that processes sensitive data, which must remain secure and not be exposed externally. What is an approach to embedding the data using Oracle Database 23ai?**
    ○ Use a third-party model via a secure API.
○ Store embeddings in an unencrypted external database.
✅ **Import and use an ONNX model.**
○ Use open-source models.

**Question 29**
**Consider the following block of code:**
vs = oracleVS(embedding_function=embed_model, client=conn23c, table_name="DEMO_TABLE", distance_strategy=DistanceStrategy.DOT_PRODUCT)
retr = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
**Which prerequisite steps must be completed before this code can execute successfully?**
    ✅ **Documents must be indexed and saved in the specified table.**
○ Embeddings must be created and stored in the database.
○ A response must be generated before running the retrieval process.
○ Documents must be retrieved from the database before running the retriever.

**Question 30**
**Which statement best describes the role of encoder and decoder models in natural language processing?**
    ○ Encoder models are used only for numerical calculations, whereas decoder models interpret them back into text.
○ Encoder models take a sequence of words and predict the next word in the sequence, whereas decoder models convert a sequence of words into a numerical representation.
✅ **Encoder models convert a sequence of words into a vector representation, and decoder models take this vector representation to generate a sequence of words.**
○ Encoder models and decoder models both convert word sequences into vectors without generating new text.

**Question 31**
**What does the output of the encoder in an encoder-decoder architecture represent?**
    ✅ **It is a sequence of embeddings that encode the semantic meaning of the input text.**
○ It is the final generated sentence ready for output by the model.
○ It represents the probabilities of the next word in the sequence.
○ It is a random initialization vector used to start the model's prediction.

**Question 32**
**Which properties must each JSON object contain in the training dataset when fine-tuning a custom model in OCI Generative AI?**
    ✅ **"prompt" and "completion"**
○ "input" and "output"
○ "request" and "response"
○ "question" and "answer"

**Question 33**
**What issue might arise from using small data sets with the Vanilla fine-tuning method in OCI Generative AI service?**
    ✅ **Overfitting**
○ Model Drift
○ Data Leakage
○ Underfitting

**Question 34**
**When should you use the T-Few fine-tuning ****method**** for training a model?**
    ✅ **For data sets with a few thousand samples or less.**
○ For models that require their own hosting dedicated AI cluster.
○ For complicated semantical understanding improvement.
○ For data sets with hundreds of thousands to millions of samples.


**Question 35**
**A data scientist is preparing a custom dataset to fine-tune an OCI Generative AI model. Which criterion must be ensured for the dataset to be accepted?**
    ✅ **The dataset must contain at least 32 prompt/completion pairs.**
○ The dataset must have a maximum of 1000 sentences per file.
○ The dataset must be in a proprietary binary format.
○ The dataset must be divided into separate files for training and validation.

**Question 36**
**How does the utilization of T-Few transformer layers contribute to the efficiency of the fine-tuning process?**
    ✅ **By restricting updates to only a specific group of transformer layers.**
○ By incorporating additional layers to the base model.
○ By allowing updates across all layers of the model.
○ By excluding transformer layers from the fine-tuning process entirely.

**Question 37**
**Which is a key advantage of using T-Few over Vanilla fine-tuning in the OCI Generative AI service?**
    ✅ **Faster training time and lower cost**
○ Reduced model complexity
○ Enhanced generalization to unseen data
○ Increased model interpretability

**Question 38**
**You need to build an LLM application using Oracle Database 23ai as the vector store and OCI Generative AI service to embed data and generate responses. What could be your approach?**
    ✅ **Use ****LangChain**** Expression Language (LCEL).**
○ Use LangChain classes to embed data outside the database and generate response.
○ Use Select AI.
○ Use DB Utils to generate embeddings and generate response using SQL.

**Question 39**
**What must be done to activate content moderation in OCI Generative AI Agents?**
    ✅ **Enable it when creating an endpoint for an agent.**
○ Configure it in the Object Storage metadata settings.
○ Enable it in the session trace settings.
○ Use a third-party content moderation API.

**Question ****40**
**How does OCI Generative AI Agents ensure that citations link to custom URLs instead of the default Object Storage links?**
    ✅ **By adding metadata to objects in Object Storage**
○ By enabling the trace feature during endpoint creation
○ By modifying the RAG agent’s retrieval mechanism
○ By increasing the session timeout for endpoints

**Question 41**
**Which of the following statements is/are applicable about Retrieval Augmented Generation (RAG)?**
    ✅ **RAG helps mitigate bias, can overcome model limitations, and can handle queries without re-training.**
○ RAG helps mitigate bias.
○ RAG can overcome model limitations.
○ RAG can handle queries without re-training.

**Question 42**
**How does the use of a vector database with RAG-based Large Language Models (LLMs) fundamentally alter their responses?**
    ✅ **It shifts the basis of their responses from static pretrained knowledge to real-time data retrieval.**
○ It transforms their architecture from a neural network to a traditional database system.
○ It enables them to bypass the need for pretraining on large text corpora.
○ It limits their ability to understand and generate natural language.
**Question 43**
**What is one of the benefits of using dedicated AI clusters in OCI Generative AI?**
    ✅ **Predictable pricing that doesn’t fluctuate with demand**
○ Unpredictable pricing that varies with demand
○ No minimum commitment required
○ A pay-per-transaction pricing model

**Question 44**
**An enterprise team deploys a hosting cluster to serve multiple versions of their fine-tuned model. They require 5 replicas for one version and 3 replicas for another. How many units will the hosting cluster require in total?**
    ✅ **8**
○ 13
○ 16
○ 11

**Question 45**
**How does the architecture of dedicated AI clusters contribute to minimizing GPU memory overhead for T-Few fine-tuned ****model**** inference?**
    ✅ **By sharing base model weights across multiple fine-tuned models on the same group of GPUs.**
○ By optimizing GPU memory utilization for each model’s unique parameters.
○ By allocating separate GPUs for each model instance.
○ By loading the entire model into GPU memory for efficient processing.

**Question 46**
**Given the prompts, classify each as employing Chain-of-Thought, Least-to-most, or Step-Back prompting.**
    ✅ **1: Chain-of-Thought, 2: Least-to-most, 3: Step-Back**
○ 1: Chain-of-Thought, 2: Step-Back, 3: Least-to-most
○ 1: Step-Back, 2: Chain-of-Thought, 3: Least-to-most
○ 1: Least-to-most, 2: Chain-of-Thought, 3: Step-Back

**Question 4****7**
**Which technique involves prompting the LLM to emit intermediate reasoning steps as part of its response?**
    ✅ **Chain-of-Thought**
○ In-context Learning
○ Step-Back Prompting
○ Least-to-most Prompting

**48. ****When should you use Prompting versus Training to achieve your goals?**
    ✅ **Use Prompting to emphasize product names in responses and Training to refine the model’s understanding of industry-specific terminology.**
○ Use Prompting to improve model’s knowledge of industry-specific terminology and Training to prioritize product names.
○ Use Prompting for both product names and terminology.
○ Use Training for both product names and terminology.

**49. Which is the main characteristic of greedy decoding in the context of language model word prediction?**
    ○ It selects words based on a flattened distribution over the vocabulary.
○ It requires a large temperature setting to ensure diverse word selection.
○ It chooses words randomly from the set of less probable candidates.
✅ **It picks the most likely word to emit at each step of decoding.**


**50. A software engineer is developing a chatbot using a large language model and must decide on a decoding strategy for generating the chatbot’s replies. Which decoding approach should they use in each of the following scenarios to achieve the desired outcome?**
    ○ For maximum consistency in the chatbot’s language, the engineer chooses greedy decoding with a low temperature setting.
✅ **To ensure the chatbot’s responses are diverse and unpredictable, the engineer sets a high temperature and uses non-deterministic decoding.**
○ In a situation requiring creative and varied responses, the engineer selects greedy decoding with an increased temperature.
○ To minimize the risk of nonsensical replies, the engineer opts for non-deterministic decoding with a very low temperature.
