**Question 1**  
**What problem can occur if there is not enough overlap between consecutive chunks when splitting a document for an LLM?**  <br>
    ○ The embeddings of the consecutive chunks may be more similar semantically.  
○ It will not increase the number of chunks of a given size.  
○ It will not have any impact.  
✅ **The continuity of the context may be lost.**  

**Question 2**  
**When using a specific LLM and splitting documents into chunks, which parameter should you check to ensure the chunks are appropriately sized for processing?**  <br>
    ✅ **Context window size.**  
○ Max number of tokens LLM can generate.  
○ Number of LLM parameters.  
○ Number of LLM layers.  

**Question 3**  
**An AI development company is working on an advanced AI assistant capable of handling queries in a seamless manner.**  
**Their goal is to create an assistant that can **analyze** images provided by users and generate descriptive text, as well as take text descriptions and produce accurate visual representations.**  <br><br>
**Considering the capabilities, which type of model would the company likely focus on integrating into their AI assistant?**  <br><br>
    ○ A language model that operates on a token-by-token output basis.  
✅ **A diffusion model that specializes in producing complex outputs.**  
○ A Retrieval-Augmented Generation (RAG) model that uses text as input and output.  
○ A Large Language Model-based agent that focuses on generating textual responses.  


**Question 4**<br>
**A student is using OCI Generative AI Embedding models to summarize long academic papers.**<br>
**If a paper exceeds the model’s token limit, but the most important insights are at the beginning, what action should the student take?**<br>
    ○ Split the paper into multiple overlapping parts and embed separately.<br>
✅ **Select to truncate the end.**<br>
○ Manually remove words before processing with embeddings.<br>
○ Select to truncate the start.<br>

**Question 5**<br>
**What is the primary function of the “temperature” parameter in OCI Generative AI Chat models?**<br><br>
    ○ Determines the maximum number of tokens the model can generate per response.<br>
○ Assigns a penalty to tokens that have already appeared in the preceding text.<br>
✅ **Controls the randomness of the model’s output, affecting its creativity.**<br>
○ Specifies a string that tells the model to stop generating more content.<br>

**Question 6**<br>
**What is the purpose of the given line of code?**<br>
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)<br>
    ✅ **It loads the OCI configuration details from a file to authenticate the client.**<br>
○ It defines the profile that will be used to generate AI models.<br>
○ It establishes a secure SSH connection to OCI services.<br>
○ It initializes a connection to the OCI Generative AI service without using authentication.<br>


**Question 7**<br>
**What is the significance of the given line of code?**<br>
chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(<br>
    model_id="ocid1.generativeaimodel.oc1.eu-frankfurt-1.amaaaaaask7dcey..."<br>
)<br>
    ○ It creates a new generative AI model instead of using an existing one.<br>
○ It sets up the storage location where AI-generated responses will be saved.<br>
✅ **It specifies the serving mode and assigns a specific generative AI model ID to be used for inference.**<br>
○ It configures a load balancer to distribute AI inference requests efficiently.<br>

**Question 8**<br>
**In the given code, what does ****setting**** truncate = "NONE" do?**<br>
embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()<br>
embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(<br>
    model_id="cohere.embed-english-v3.0"<br>
)<br>
embed_text_detail.inputs = inputs<br>
embed_text_detail.truncate = "NONE"<br>
    ✅ **It prevents input text from being truncated before processing.**<br>
○ It ensures that only a single word from the input is used for embedding.<br>
○ It forces the model to limit the output text length.<br>
○ It removes all white space from the input text.<br>


**Question 9**
**What distinguishes the Cohere Embed v3 model from its predecessor in the OCI Generative AI service?**<br>
    ○ Support for tokenizing longer sentences<br>
○ Emphasis on syntactic clustering of word embeddings<br>
○ Capacity to translate text in over 20 languages<br>
✅ **Improved retrievals for Retrieval-Augmented Generation (RAG) systems**<br>

**Question 10**<br>
**Which of the following statements is NOT true?**
    ✅ **Embeddings are represented as single-dimensional numerical values that capture text meaning.**
○ Embeddings can be used to compare text based on semantic similarity.
○ Embeddings of sentences with similar meanings are positioned close to each other in vector space.
○ Embeddings can be created for words, sentences and entire documents.

**Question 11**
**In OCI Generative AI Agents, what happens if a session-enabled endpoint remains idle for the specified timeout period?**<br>
○ The session remains active indefinitely until manually ended.<br>
○ The session restarts and retains the previous context.<br>
○ The agent deletes all data related to the session.<br>
✅ **The session automatically ends and subsequent conversations do not retain the previous context.**<br>

**Question 12**
**What happens when you enable the session option while creating an endpoint in Generative AI Agents?**<br>
✅**The context of the chat session is retained, and the option cannot be changed later.****<br>
**○ The agent stops responding after one hour of inactivity.<br>
○ The context of the chat session is retained, but the option can be disabled later.<br>
○ All conversations are saved permanently regardless of session settings.<br>

**Question 13**<br>
**Which option is available when moving an endpoint resource to a different compartment in Generative AI Agents?**<br>
○ Create a duplicate endpoint in the new compartment manually.<br>
○ Modify the endpoint's data source to match the new compartment.<br>
✅ **Select a new compartment for the endpoint and move the resource.**<br>
○ Archive the endpoint before moving it to a new compartment.<br>

**Question 14**<br>
**What happens when you delete a knowledge base in OCI Generative AI Agents?**<br>
○ The knowledge base is archived for later recovery.<br>
✅ **The knowledge base is permanently deleted, and the action cannot be undone.**<br>
○ Only the metadata of the knowledge base is removed.<br>
○ The knowledge base is marked inactive but remains stored in the system.<br>

**Question 15**<br>
**You have set up an Oracle Database 23c AI table so that Generative AI Agents can connect to it. You now need to set up a database function that can return vector search results from each query.****<br>
What does the score field represent in the vector search results returned by the database function?**<br>
○ The top_k rank of the document in the search results<br>
○ The token count of the BODY content<br>
✅ **The distance between the query vector and the BODY vector.**<br>
○ The unique identifier for each document.<br>

**Question 16**<br>
**In OCI Generative AI Agents, if an ingestion job processes 20 files and 2 fail, what happens when the job is restarted?**<br>
✅ **Only the 2 failed files that have been updated are ingested.**<br>
○ All 20 files are re-ingested from the beginning.<br>
○ The job processes all 20 files regardless of updates.<br>
○ None of the files are processed during the restart.<br>


**Question 17**<br>
**How should you handle a data source in OCI Generative AI Agents if your data is not ready yet?**<br>
✅ **Create an empty folder for the data source and populate it later.**<br>
○ Leave the data source configuration incomplete until the data is ready.<br>
○ Upload placeholder files larger than 100 MB as a temporary solution.<br>
○ Use multiple buckets to store the incomplete data.<br>

**Question 18**<br>
**How are fine-tuned customer models stored to enable strong data privacy and security in OCI Generative AI service?**<br>
✅ **Stored in OCI Object Storage and encrypted by default.****<br>
**○ Stored in OCI Key Management service.<br>
○ Stored in an unencrypted form in OCI Object Storage.<br>
○ Shared among multiple customers for efficiency.<br>

**Question 19**<br>
**How long does the OCI Generative AI Agents service retain customer****‐****provided queries and retrieved context?**<br>
✅ **Only during the user’s session.**<br>
○ Indefinitely, for future analysis.<br>
○ Until the customer deletes the data manually.<br>
○ For up to 30 days after the session ends.<br>

**Question 20**<br>
**What does a dedicated RDMA cluster network do during model fine-tuning and inference?**<br>
    ○ It increases GPU memory requirements for model deployment.<br>
○ It leads to higher latency in model inference.<br>
✅ **It enables the deployment of multiple fine-tuned models within a single cluster.**<br>
○ It limits the number of fine-tuned models deployable on the same GPU cluster.<br>


**Question 21**<br>
**Which role does a “model endpoint” serve in the inference workflow of the OCI Generative AI service?**<br>
    ○ Evaluates the performance metrics of the custom models.<br>
○ Hosts the training data for fine-tuning custom models.<br>
✅ **Serves as a designated point for user requests and model responses.**<br>
○ Updates the weights of the base model during the fine-tuning process.<br>

**Question 22**<br>
**A startup is evaluating the cost implications of using the OCI Generative AI service for an app that generates text. They anticipate a steady but moderate volume of requests. Which pricing model would be most appropriate for them?**<br>
    ○ Dedicated AI clusters, as they offer a fixed monthly rate regardless of usage.<br>
○ On-demand inferencing, as it provides a flat fee for unlimited usage.<br>
○ Dedicated AI clusters, as they are mandatory for any text-generation tasks.<br>
✅ **On-demand inferencing, as it allows them to pay per character processed without long-term commitments.**<br>

**Question 23**<br>
**How do Dot Product and Cosine Distance differ in their application to comparing text embeddings in NLP?**<br>
    ✅ **Dot Product measures the magnitude and direction of vectors, whereas Cosine Distance focuses on the orientation regardless of magnitude.**<br>
○ Dot Product is used for semantic analysis, whereas Cosine Distance is used for syntactic comparisons.<br>
○ Dot Product calculates the literal overlap of words, whereas Cosine Distance evaluates the stylistic similarity.<br>
○ Dot Product assesses overall similarity in content, whereas Cosine Distance measures topical relevance.<br>


**Question 24**<br>
**How does Retrieval-Augmented Generation (RAG) differ from prompt engineering and fine-tuning in terms of setup complexity?**<br>
     ○ RAG is simpler to implement as it does not require training costs.<br>
✅ **RAG is more complex to set up and requires a compatible data source.**<br>
○ RAG involves adding LLM optimization to the model’s prompt.<br>
○ RAG requires fine-tuning on a smaller domain-specific dataset.<br>

**Question 25**<br>
**Which is a distinguishing feature of “Parameter-Efficient Fine-Tuning (PEFT)” as opposed to classic fine-tuning in Large Language Model training?**<br>
    ✅ **PEFT involves only a few or new parameters and uses ****labeled****, task-specific data.**<br>
○ PEFT modifies all parameters and uses unlabeled, task-agnostic data.<br>
○ PEFT modifies all parameters and is typically used when no training data exists.<br>
○ PEFT does not modify any parameters but uses soft prompting with unlabeled data.<br>

**Question 26**<br>
**How can you verify that an LLM-generated response is grounded in factual and relevant information?**<br>
    ○ Examine the document chunks stored in the vector database.<br>
✅ **Check the references to the documents provided in the response.**<br>
○ Use model evaluators to assess the accuracy and relevance of responses.<br>
○ Manually review past conversations to ensure consistency in responses.<br>

**Question 27**<br>
**Which category of pretrained foundational models is available for on-demand serving mode in the OCI Generative AI service?**<br>
    ○ Generation Models<br>
○ Summarization Models<br>
✅ **Chat Models**<br>
○ Translation Models<br>

**Question 28**<br>
**You are developing a chatbot that processes sensitive data, which must remain secure and not be exposed externally. What is an approach to embedding the data using Oracle Database 23ai?**<br>
    ○ Use a third-party model via a secure API.<br>
○ Store embeddings in an unencrypted external database.<br>
✅ **Import and use an ONNX model.**
○ Use open-source models.<br>

**Question 29**<br>
**Consider the following block of code:**<br>
vs = oracleVS(embedding_function=embed_model, client=conn23c, table_name="DEMO_TABLE", distance_strategy=DistanceStrategy.DOT_PRODUCT)<br>
retr = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})<br>
**Which prerequisite steps must be completed before this code can execute successfully?**<br>
    ✅ **Documents must be indexed and saved in the specified table.**<br>
○ Embeddings must be created and stored in the database.<br>
○ A response must be generated before running the retrieval process.<br>
○ Documents must be retrieved from the database before running the retriever.<br>

**Question 30**<br>
**Which statement best describes the role of encoder and decoder models in natural language processing?**<br>
    ○ Encoder models are used only for numerical calculations, whereas decoder models interpret them back into text.<br>
○ Encoder models take a sequence of words and predict the next word in the sequence, whereas decoder models convert a sequence of words into a numerical representation.<br>
✅ **Encoder models convert a sequence of words into a vector representation, and decoder models take this vector representation to generate a sequence of words.**<br>
○ Encoder models and decoder models both convert word sequences into vectors without generating new text.<br>

**Question 31**<br>
**What does the output of the encoder in an encoder-decoder architecture represent?**<br>
    ✅ **It is a sequence of embeddings that encode the semantic meaning of the input text.**<br>
○ It is the final generated sentence ready for output by the model.<br>
○ It represents the probabilities of the next word in the sequence.<br>
○ It is a random initialization vector used to start the model's prediction.<br>

**Question 32**<br>
**Which properties must each JSON object contain in the training dataset when fine-tuning a custom model in OCI Generative AI?**<br>
    ✅ **"prompt" and "completion"**<br>
○ "input" and "output"<br>
○ "request" and "response"<br>
○ "question" and "answer"<br>

**Question 33**<br>
**What issue might arise from using small data sets with the Vanilla fine-tuning method in OCI Generative AI service?**<br>
    ✅ **Overfitting**<br>
○ Model Drift<br>
○ Data Leakage<br>
○ Underfitting<br>

**Question 34**
**When should you use the T-Few fine-tuning method for training a model?**<br>
    ✅ **For data sets with a few thousand samples or less.**<br>
○ For models that require their own hosting dedicated AI cluster.<br>
○ For complicated semantical understanding improvement.<br>
○ For data sets with hundreds of thousands to millions of samples.<br>


**Question 35**<br>
**A data scientist is preparing a custom dataset to fine-tune an OCI Generative AI model. Which criterion must be ensured for the dataset to be accepted?**<br>
    ✅ **The dataset must contain at least 32 prompt/completion pairs.**<br>
○ The dataset must have a maximum of 1000 sentences per file.<br>
○ The dataset must be in a proprietary binary format.<br>
○ The dataset must be divided into separate files for training and validation.<br>

**Question 36**<br>
**How does the utilization of T-Few transformer layers contribute to the efficiency of the fine-tuning process?**<br>
    ✅ **By restricting updates to only a specific group of transformer layers.**<br>
○ By incorporating additional layers to the base model.<br>
○ By allowing updates across all layers of the model.<br>
○ By excluding transformer layers from the fine-tuning process entirely.<br>

**Question 37**<br>
**Which is a key advantage of using T-Few over Vanilla fine-tuning in the OCI Generative AI service?**<br>
    ✅ **Faster training time and lower cost**<br>
○ Reduced model complexity<br>
○ Enhanced generalization to unseen data<br>
○ Increased model interpretability<br>

**Question 38**
**You need to build an LLM application using Oracle Database 23ai as the vector store and OCI Generative AI service to embed data and generate responses. What could be your approach?**
    ✅ **Use ****LangChain**** Expression Language (LCEL).**
○ Use LangChain classes to embed data outside the database and generate response.
○ Use Select AI.
○ Use DB Utils to generate embeddings and generate response using SQL.

**Question 39**<br>
**What must be done to activate content moderation in OCI Generative AI Agents?**<br>
    ✅ **Enable it when creating an endpoint for an agent.**<br>
○ Configure it in the Object Storage metadata settings.<br>
○ Enable it in the session trace settings.<br>
○ Use a third-party content moderation API.<br>

**Question 40**<br>
**How does OCI Generative AI Agents ensure that citations link to custom URLs instead of the default Object Storage links?**<br>
    ✅ **By adding metadata to objects in Object Storage**<br>
○ By enabling the trace feature during endpoint creation<br>
○ By modifying the RAG agent’s retrieval mechanism<br>
○ By increasing the session timeout for endpoints<br>

**Question 41**<br>
**Which of the following statements is/are applicable about Retrieval Augmented Generation (RAG)?**<br>
    ✅ **RAG helps mitigate bias, can overcome model limitations, and can handle queries without re-training.**<br>
○ RAG helps mitigate bias.<br>
○ RAG can overcome model limitations.<br>
○ RAG can handle queries without re-training.<br>

**Question 42**<br>
**How does the use of a vector database with RAG-based Large Language Models (LLMs) fundamentally alter their responses?**<br>
    ✅ **It shifts the basis of their responses from static pretrained knowledge to real-time data retrieval.**<br>
○ It transforms their architecture from a neural network to a traditional database system.<br>
○ It enables them to bypass the need for pretraining on large text corpora.<br>
○ It limits their ability to understand and generate natural language.<br>

**Question 43**<br>
**What is one of the benefits of using dedicated AI clusters in OCI Generative AI?**<br>
    ✅ **Predictable pricing that doesn’t fluctuate with demand**<br>
○ Unpredictable pricing that varies with demand<br>
○ No minimum commitment required<br>
○ A pay-per-transaction pricing model<br>

**Question 44**<br>
**An enterprise team deploys a hosting cluster to serve multiple versions of their fine-tuned model. They require 5 replicas for one version and 3 replicas for another. How many units will the hosting cluster require in total?**<br>
    ✅ **8**<br>
○ 13<br>
○ 16<br>
○ 11<br>

**Question 45**<br>
**How does the architecture of dedicated AI clusters contribute to minimizing GPU memory overhead for T-Few fine-tuned **model** inference?**<br>
    ✅ **By sharing base model weights across multiple fine-tuned models on the same group of GPUs.**<br>
○ By optimizing GPU memory utilization for each model’s unique parameters.<br>
○ By allocating separate GPUs for each model instance.<br>
○ By loading the entire model into GPU memory for efficient processing.<br>

**Question 46**<br>
**Given the prompts, classify each as employing Chain-of-Thought, Least-to-most, or Step-Back prompting.**<br>
Prompt 1<br>
“Calculate the total number of wheels needed for 3 cars. Cars have 4 wheels each. Then, use the total number of wheels to determine how many sets of wheels we can buy with $200 if one set (4 wheels) costs $50.”<br><br>
Prompt 2<br>

“Solve a complex math problem by first identifying the formula needed, and then solve a simpler version of the problem before tackling the full question.”<br><br>
Prompt 3<br>

“To understand the impact of greenhouse gases on climate change, let’s start by defining what greenhouse gases are. Next, we’ll explore how they trap heat in the Earth’s atmosphere.”<br>
    ✅ **1: Chain-of-Thought, 2: Least-to-most, 3: Step-Back**<br>
○ 1: Chain-of-Thought, 2: Step-Back, 3: Least-to-most<br>
○ 1: Step-Back, 2: Chain-of-Thought, 3: Least-to-most<br>
○ 1: Least-to-most, 2: Chain-of-Thought, 3: Step-Back<br>

**Question 47**<br>
**Which technique involves prompting the LLM to emit intermediate reasoning steps as part of its response?**<br>
    ✅ **Chain-of-Thought**<br>
○ In-context Learning<br>
○ Step-Back Prompting<br>
○ Least-to-most Prompting<br>

**48. When should you use Prompting versus Training to achieve your goals?**<br>
    ✅ **Use Prompting to emphasize product names in responses and Training to refine the model’s understanding of industry-specific terminology.**<br>
○ Use Prompting to improve model’s knowledge of industry-specific terminology and Training to prioritize product names.<br>
○ Use Prompting for both product names and terminology.<br>
○ Use Training for both product names and terminology.<br>

**49. Which is the main characteristic of greedy decoding in the context of language model word prediction?**<br>
    ○ It selects words based on a flattened distribution over the vocabulary.<br>
○ It requires a large temperature setting to ensure diverse word selection.<br>
○ It chooses words randomly from the set of less probable candidates.<br>
✅ **It picks the most likely word to emit at each step of decoding.**<br>


**50. A software engineer is developing a chatbot using a large language model and must decide on a decoding strategy for generating the chatbot’s replies.<br> Which decoding approach should they use in each of the following scenarios to achieve the desired outcome?**<br><br>
    ○ For maximum consistency in the chatbot’s language, the engineer chooses greedy decoding with a low temperature setting.<br>
✅ **To ensure the chatbot’s responses are diverse and unpredictable, the engineer sets a high temperature and uses non-deterministic decoding.**<br>
○ In a situation requiring creative and varied responses, the engineer selects greedy decoding with an increased temperature.<br>
○ To minimize the risk of nonsensical replies, the engineer opts for non-deterministic decoding with a very low temperature.<br>







