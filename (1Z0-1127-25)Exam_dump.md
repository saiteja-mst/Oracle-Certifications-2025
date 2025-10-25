# Oracle Exam 1Z0-1127-25 : Oracle Cloud Infrastructure 2025 Generative AI Professional

#### Q1. What problem can occur if there is not enough overlap between consecutive chunks when splitting a document for an LLM?
- [ ] The embeddings of the consecutive chunks may be more similar semantically.
- [ ] It will not increase the number of chunks of a given size.
- [ ] It will not have any impact.
- [x] The continuity of the context may be lost. ✅

#### Q2. When using a specific LLM and splitting documents into chunks, which parameter should you check to ensure the chunks are appropriately sized for processing?
- [x] Context window size. ✅
- [ ] Max number of tokens LLM can generate.
- [ ] Number of LLM parameters.
- [ ] Number of LLM layers.

#### Q3. Considering the capabilities, which type of model would the company likely focus on integrating into their AI assistant?
- [ ] A language model that operates on a token-by-token output basis.
- [x] A diffusion model that specializes in producing complex outputs. ✅
- [ ] A Retrieval-Augmented Generation (RAG) model that uses text as input and output.
- [ ] A Large Language Model-based agent that focuses on generating textual responses.

#### Q4. If a paper exceeds the model’s token limit, but the most important insights are at the beginning, what action should the student take?
- [ ] Split the paper into multiple overlapping parts and embed separately.
- [x] Select to truncate the end. ✅
- [ ] Manually remove words before processing with embeddings.
- [ ] Select to truncate the start.

#### Q5. What is the primary function of the “temperature” parameter in OCI Generative AI Chat models?
- [ ] Determines the maximum number of tokens the model can generate per response.
- [ ] Assigns a penalty to tokens that have already appeared in the preceding text.
- [x] Controls the randomness of the model’s output, affecting its creativity. ✅
- [ ] Specifies a string that tells the model to stop generating more content.

#### Q6. What is the purpose of the given line of code?
`config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)`
- [x] It loads the OCI configuration details from a file to authenticate the client. ✅
- [ ] It defines the profile that will be used to generate AI models.
- [ ] It establishes a secure SSH connection to OCI services.
- [ ] It initializes a connection to the OCI Generative AI service without using authentication.

#### Q7. What is the significance of the given line of code?
`chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id="ocid1.generativeaimodel.oc1.eu-frankfurt-1.amaaaaaask7dcey...")`
- [ ] It creates a new generative AI model instead of using an existing one.
- [ ] It sets up the storage location where AI-generated responses will be saved.
- [x] It specifies the serving mode and assigns a specific generative AI model ID to be used for inference. ✅
- [ ] It configures a load balancer to distribute AI inference requests efficiently.

#### Q8. In the given code, what does setting `truncate = "NONE"` do?
- [x] It prevents input text from being truncated before processing. ✅
- [ ] It ensures that only a single word from the input is used for embedding.
- [ ] It forces the model to limit the output text length.
- [ ] It removes all white space from the input text.

#### Q9. What distinguishes the Cohere Embed v3 model from its predecessor in the OCI Generative AI service?
- [ ] Support for tokenizing longer sentences
- [ ] Emphasis on syntactic clustering of word embeddings
- [ ] Capacity to translate text in over 20 languages
- [x] Improved retrievals for Retrieval-Augmented Generation (RAG) systems ✅

#### Q10. Which of the following statements is NOT true?
- [x] Embeddings are represented as single-dimensional numerical values that capture text meaning. ✅
- [ ] Embeddings can be used to compare text based on semantic similarity.
- [ ] Embeddings of sentences with similar meanings are positioned close to each other in vector space.
- [ ] Embeddings can be created for words, sentences and entire documents.

#### Q11. In OCI Generative AI Agents, what happens if a session-enabled endpoint remains idle for the specified timeout period?
- [ ] The session remains active indefinitely until manually ended.
- [ ] The session restarts and retains the previous context.
- [ ] The agent deletes all data related to the session.
- [x] The session automatically ends and subsequent conversations do not retain the previous context. ✅

#### Q12. What happens when you enable the session option while creating an endpoint in Generative AI Agents?
- [x] The context of the chat session is retained, and the option cannot be changed later. ✅
- [ ] The agent stops responding after one hour of inactivity.
- [ ] The context of the chat session is retained, but the option can be disabled later.
- [ ] All conversations are saved permanently regardless of session settings.

#### Q13. Which option is available when moving an endpoint resource to a different compartment in Generative AI Agents?
- [ ] Create a duplicate endpoint in the new compartment manually.
- [ ] Modify the endpoint's data source to match the new compartment.
- [x] Select a new compartment for the endpoint and move the resource. ✅
- [ ] Archive the endpoint before moving it to a new compartment.

#### Q14. What happens when you delete a knowledge base in OCI Generative AI Agents?
- [ ] The knowledge base is archived for later recovery.
- [x] The knowledge base is permanently deleted, and the action cannot be undone. ✅
- [ ] Only the metadata of the knowledge base is removed.
- [ ] The knowledge base is marked inactive but remains stored in the system.

#### Q15. What does the `score` field represent in the vector search results returned by the database function?
- [ ] The top_k rank of the document in the search results.
- [ ] The token count of the BODY content.
- [x] The distance between the query vector and the BODY vector. ✅
- [ ] The unique identifier for each document.

#### Q16. In OCI Generative AI Agents, if an ingestion job processes 20 files and 2 fail, what happens when the job is restarted?
- [x] Only the 2 failed files that have been updated are ingested. ✅
- [ ] All 20 files are re-ingested from the beginning.
- [ ] The job processes all 20 files regardless of updates.
- [ ] None of the files are processed during the restart.

#### Q17. How should you handle a data source in OCI Generative AI Agents if your data is not ready yet?
- [x] Create an empty folder for the data source and populate it later. ✅
- [ ] Leave the data source configuration incomplete until the data is ready.
- [ ] Upload placeholder files larger than 100 MB as a temporary solution.
- [ ] Use multiple buckets to store the incomplete data.

#### Q18. How are fine-tuned customer models stored to enable strong data privacy and security in OCI Generative AI service?
- [x] Stored in OCI Object Storage and encrypted by default. ✅
- [ ] Stored in OCI Key Management service.
- [ ] Stored in an unencrypted form in OCI Object Storage.
- [ ] Shared among multiple customers for efficiency.

#### Q19. How long does the OCI Generative AI Agents service retain customer-provided queries and retrieved context?
- [x] Only during the user’s session. ✅
- [ ] Indefinitely, for future analysis.
- [ ] Until the customer deletes the data manually.
- [ ] For up to 30 days after the session ends.

#### Q20. What does a dedicated RDMA cluster network do during model fine-tuning and inference?
- [ ] It increases GPU memory requirements for model deployment.
- [ ] It leads to higher latency in model inference.
- [x] It enables the deployment of multiple fine-tuned models within a single cluster. ✅
- [ ] It limits the number of fine-tuned models deployable on the same GPU cluster.

#### Q21. Which role does a “model endpoint” serve in the inference workflow of the OCI Generative AI service?
- [ ] Evaluates the performance metrics of the custom models.
- [ ] Hosts the training data for fine-tuning custom models.
- [x] Serves as a designated point for user requests and model responses. ✅
- [ ] Updates the weights of the base model during the fine-tuning process.

#### Q22. A startup anticipates a steady but moderate volume of text-generation requests. Which pricing model is most appropriate?
- [ ] Dedicated AI clusters, as they offer a fixed monthly rate regardless of usage.
- [ ] On-demand inferencing, as it provides a flat fee for unlimited usage.
- [ ] Dedicated AI clusters, as they are mandatory for text-generation tasks.
- [x] On-demand inferencing, as it allows pay-per-character processing without long-term commitments. ✅

#### Q23. How do Dot Product and Cosine Distance differ in comparing text embeddings in NLP?
- [x] Dot Product measures magnitude and direction, whereas Cosine Distance focuses on orientation regardless of magnitude. ✅
- [ ] Dot Product is used for semantic analysis, whereas Cosine Distance is used for syntactic comparisons.
- [ ] Dot Product calculates literal overlap of words, whereas Cosine Distance evaluates stylistic similarity.
- [ ] Dot Product assesses overall similarity in content, whereas Cosine Distance measures topical relevance.

#### Q24. How does Retrieval-Augmented Generation (RAG) differ from prompt engineering and fine-tuning in setup complexity?
- [ ] RAG is simpler to implement as it does not require training costs.
- [x] RAG is more complex to set up and requires a compatible data source. ✅
- [ ] RAG involves adding LLM optimization to the model’s prompt.
- [ ] RAG requires fine-tuning on a smaller domain-specific dataset.

#### Q25. Which is a distinguishing feature of Parameter-Efficient Fine-Tuning (PEFT)?
- [x] PEFT involves only a few or new parameters and uses labeled, task-specific data. ✅
- [ ] PEFT modifies all parameters and uses unlabeled, task-agnostic data.
- [ ] PEFT modifies all parameters and is used when no training data exists.
- [ ] PEFT does not modify any parameters but uses soft prompting with unlabeled data.

#### Q26. How can you verify that an LLM-generated response is grounded in factual information?
- [ ] Examine document chunks stored in the vector database.
- [x] Check the references to the documents provided in the response. ✅
- [ ] Use model evaluators to assess accuracy.
- [ ] Manually review past conversations for consistency.

#### Q27. Which category of pretrained foundational models is available for on-demand serving mode in OCI Generative AI?
- [ ] Generation Models
- [ ] Summarization Models
- [x] Chat Models ✅
- [ ] Translation Models

#### Q28. What is a secure approach to embedding sensitive data using Oracle Database 23ai?
- [ ] Use a third-party model via a secure API.
- [ ] Store embeddings in an unencrypted external database.
- [x] Import and use an ONNX model. ✅
- [ ] Use open-source models.

#### Q29. Which prerequisite must be completed before executing the following code?
`vs = oracleVS(embedding_function=embed_model, client=conn23c, table_name="DEMO_TABLE", distance_strategy=DistanceStrategy.DOT_PRODUCT)`
`retr = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})`
- [x] Documents must be indexed and saved in the specified table. ✅
- [ ] Embeddings must be created and stored in the database.
- [ ] A response must be generated before running the retriever.
- [ ] Documents must be retrieved before running the retriever.

#### Q30. Which best describes the role of encoder and decoder models in NLP?
- [ ] Encoder models are used only for numerical calculations, while decoders interpret them back into text.
- [ ] Encoders take word sequences and predict the next word; decoders convert text to numerical form.
- [x] Encoders convert text to vector representations; decoders use those vectors to generate text sequences. ✅
- [ ] Both encoders and decoders convert text into vectors without generating new text.

#### Q31. What does the output of the encoder in an encoder-decoder architecture represent?
- [x] A sequence of embeddings encoding the semantic meaning of the input text. ✅
- [ ] The final generated sentence ready for output.
- [ ] The probabilities of the next word in the sequence.
- [ ] A random initialization vector to start predictions.

#### Q32. Which properties must each JSON object contain when fine-tuning a custom model in OCI Generative AI?
- [x] "prompt" and "completion" ✅
- [ ] "input" and "output"
- [ ] "request" and "response"
- [ ] "question" and "answer"

#### Q33. What issue might arise from using small datasets with Vanilla fine-tuning?
- [x] Overfitting ✅
- [ ] Model Drift
- [ ] Data Leakage
- [ ] Underfitting

#### Q34. When should you use the T-Few fine-tuning method?
- [x] For datasets with a few thousand samples or less. ✅
- [ ] For models requiring dedicated AI clusters.
- [ ] For complex semantic understanding improvement.
- [ ] For datasets with hundreds of thousands to millions of samples.

#### Q35. What criterion must be met for a dataset to be accepted when fine-tuning in OCI Generative AI?
- [x] Must contain at least 32 prompt/completion pairs. ✅
- [ ] Maximum of 1000 sentences per file.
- [ ] Must be in proprietary binary format.
- [ ] Must be divided into separate training and validation files.

#### Q36. How does using T-Few transformer layers improve fine-tuning efficiency?
- [x] By restricting updates to a specific group of transformer layers. ✅
- [ ] By adding extra layers to the base model.
- [ ] By allowing updates across all model layers.
- [ ] By excluding transformer layers entirely.

#### Q37. What is a key advantage of using T-Few over Vanilla fine-tuning?
- [x] Faster training time and lower cost. ✅
- [ ] Reduced model complexity.
- [ ] Enhanced generalization to unseen data.
- [ ] Increased interpretability.

#### Q38. How can you build an LLM app using Oracle Database 23ai and OCI Generative AI?
- [x] Use LangChain Expression Language (LCEL). ✅
- [ ] Use LangChain classes to embed data outside the DB.
- [ ] Use Select AI.
- [ ] Use DB Utils with SQL for embeddings.

#### Q39. What must be done to activate content moderation in OCI Generative AI Agents?
- [x] Enable it when creating an endpoint for an agent. ✅
- [ ] Configure it in Object Storage metadata.
- [ ] Enable it in session trace settings.
- [ ] Use a third-party moderation API.

#### Q40. How does OCI Generative AI ensure citations link to custom URLs?
- [x] By adding metadata to objects in Object Storage. ✅
- [ ] By enabling trace feature during endpoint creation.
- [ ] By modifying the RAG retrieval mechanism.
- [ ] By increasing session timeout.

#### Q41. Which statements apply to Retrieval-Augmented Generation (RAG)?
- [x] RAG helps mitigate bias, overcome model limits, and handle queries without re-training. ✅
- [ ] RAG helps mitigate bias.
- [ ] RAG can overcome model limitations.
- [ ] RAG can handle queries without retraining.

#### Q42. How does a vector database alter RAG-based LLM responses?
- [x] It shifts responses from static pretrained knowledge to real-time retrieval. ✅
- [ ] It converts the model into a traditional database system.
- [ ] It removes the need for pretraining.
- [ ] It limits natural language understanding.

#### Q43. What is one benefit of using dedicated AI clusters in OCI Generative AI?
- [x] Predictable pricing that doesn’t fluctuate with demand. ✅
- [ ] Unpredictable pricing varying with demand.
- [ ] No minimum commitment.
- [ ] Pay-per-transaction model.

#### Q44. If a team deploys 5 replicas for one model and 3 for another, how many units are required?
- [x] 8 ✅
- [ ] 13
- [ ] 16
- [ ] 11

#### Q45. How do dedicated AI clusters minimize GPU memory overhead for T-Few inference?
- [x] By sharing base model weights across fine-tuned models on the same GPUs. ✅
- [ ] By optimizing GPU memory for each model’s parameters.
- [ ] By allocating separate GPUs per model.
- [ ] By loading entire models into GPU memory.

#### Q46. Given the prompts, classify them as Chain-of-Thought, Least-to-most, or Step-Back prompting.
- [x] 1: Chain-of-Thought, 2: Least-to-most, 3: Step-Back ✅
- [ ] 1: Chain-of-Thought, 2: Step-Back, 3: Least-to-most
- [ ] 1: Step-Back, 2: Chain-of-Thought, 3: Least-to-most
- [ ] 1: Least-to-most, 2: Chain-of-Thought, 3: Step-Back

#### Q47. Which technique involves prompting the LLM to emit intermediate reasoning steps?
- [x] Chain-of-Thought ✅
- [ ] In-context Learning
- [ ] Step-Back Prompting
- [ ] Least-to-most Prompting

#### Q48. When should you use Prompting vs Training?
- [x] Use Prompting to emphasize product names; Training to refine industry terminology. ✅
- [ ] Use Prompting to improve terminology; Training to prioritize product names.
- [ ] Use Prompting for both names and terminology.
- [ ] Use Training for both names and terminology.

#### Q49. What is the main characteristic of greedy decoding in LLMs?
- [ ] It selects words based on a flattened vocabulary distribution.
- [ ] It requires a high temperature for diverse outputs.
- [ ] It picks random low-probability words.
- [x] It picks the most likely word at each decoding step. ✅

#### Q50. Which decoding strategy should be used for different chatbot outcomes?
- [ ] For maximum consistency, use greedy decoding with low temperature.
- [x] For diverse, unpredictable replies, use high temperature and non-deterministic decoding. ✅
- [ ] For creativity, use greedy decoding with increased temperature.
- [ ] To avoid nonsense replies, use non-deterministic decoding with very low temperature.

