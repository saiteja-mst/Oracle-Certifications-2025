# Oracle Cloud Infrastructure 2025 \- Generative AI Professional Practice Exam, you can find the answers here -> [Solutions](#answers)

#### Q1. What is the role of the inputs parameter in the given code snippet?
inputs = [
"Learn about the Employee Stock Purchase Plan",
"Reassign timecard approvals during leave",
"View my payslip online",
]
embed_text_detail.inputs = inputs
- [ ] It sets the output format for the embeddings.
- [ ] It controls the maximum number of embeddings the model can generate.
- [ ] It provides metadata about the embedding process.
- [x] It specifies the text data that will be converted into embeddings. ✅

#### Q2. What must be done before you can delete a knowledge base in Generative AI Agents?
- [ ] Disconnect the database tool connection.
- [ ] Archive the knowledge base for future use.
- [ ] Reassign the knowledge base to a different agent.
- [x] Delete the data sources and agents using that knowledge base. ✅

#### Q3. A data scientist… They notice loss decreasing. What does the loss metric indicate?
- [ ] Loss measures the total number of predictions made by the model during training.
- [ ] Loss only evaluates the accuracy of correct predictions, ignoring the impact of incorrect predictions.
- [x] Loss quantifies how far the model's predictions deviate from the actual values, indicating how wrong the predictions are. ✅
- [ ] Loss reflects the quality of predictions and should increase as the model improves.

#### Q4. What does accuracy measure in the context of fine-tuning results for a generative model?
- [ ] The proportion of incorrect predictions made by the model during an evaluation.
- [ ] The depth of the neural network layers used in the model.
- [x] How many predictions the model made correctly out of all the predictions in an evaluation. ✅
- [ ] The number of predictions a model makes, regardless of whether they are correct or incorrect.

#### Q5. What is the purpose of the VECTOR field in the Oracle Database 23ai table for Generative AI Agents?
- [ ] To store the document TITLE.
- [ ] To store the URL references for the documents.
- [x] To store the embeddings generated from the BODY content. ✅
- [ ] To assign a unique identifier DOCID to each document.

#### Q6. Why are diffusion models difficult to apply to text generation tasks?
- [ ] Because diffusion models can only produce images.
- [x] Because text representation is categorical, unlike images. ✅
- [ ] Because text generation does not require complex models.
- [ ] Because text is not categorical.

#### Q7. In the simplified workflow for managing and querying vector data, what is the role of indexing?
- [x] Mapping vectors to a data structure for faster searching, enabling efficient retrieval. ✅
- [ ] Compressing vector data for minimized storage usage.
- [ ] Categorizing vectors based on their originating data type (text, images, audio).
- [ ] Converting vectors into a non-indexed format for easier retrieval.

#### Q8. How does a presence penalty function when using OCI Generative AI chat models?
- [ ] It applies a penalty only if the token has appeared more than twice.
- [ ] It only penalizes tokens that have never appeared in the text before.
- [x] It penalizes a token each time it appears after the first occurrence. ✅
- [ ] It penalizes all tokens equally, regardless of how often they have appeared.

#### Q9. Fine-tuning multiple models on the same cluster… total units provisioned?
- [ ] 2
- [ ] 8
- [ ] 6
- [x] 1 ✅

#### Q10. Accuracy in vector databases contributes to LLMs by preserving which relationships and why?
- [x] Semantic relationships, and they are crucial for understanding context and generating precise language. ✅
- [ ] Linear relationships, and they simplify the modeling process.
- [ ] Hierarchical relationships, and they are important for structuring database queries.
- [ ] Temporal relationships, and they are necessary for predicting future linguistic trends.

#### Q11. How does the temperature setting influence the probability distribution over the vocabulary?
- [ ] Temperature has no effect on the probability distribution; it only changes the speed of decoding.
- [ ] Decreasing temperature broadens the distribution, making less likely words more probable.
- [x] Increasing temperature flattens the distribution, allowing for more varied word choices. ✅
- [ ] Increasing temperature removes the impact of the most likely word.

#### Q12. Which component of RAG evaluates and prioritizes the retrieved information?
- [ ] Encoder-decoder
- [x] Ranker ✅
- [ ] Retriever
- [ ] Generator

#### Q13. Destination port range in the subnet's ingress rule for an Oracle Database?
- [ ] 8080–8081
- [ ] 1433–1434
- [ ] 3306–3307
- [x] 1521–1522 ✅

#### Q14. Which statement is true about the "Top p" parameter?
- [ ] "Top p" selects tokens from the "top k" tokens sorted by probability.
- [ ] "Top p" assigns penalties to frequently occurring tokens.
- [ ] "Top p" determines the maximum number of tokens per response.
- [x] "Top p" limits token selection based on the sum of their probabilities. ✅

#### Q15. Model behavior if you don't provide a value for the seed parameter?
- [ ] The model generates responses deterministically.
- [ ] The model restricts the maximum number of tokens that can be generated.
- [ ] The model assigns a default seed value of 9999.
- [x] The model gives diverse responses. ✅

#### Q16. How many dedicated AI clusters are required to host at least 60 endpoints?
- [ ] 1
- [ ] 5
- [ ] 2
- [x] 3 ✅

#### Q17. Which statement is true about RAG?
- [ ] It is solely used in QA-based scenarios.
- [x] It is non-parametric and can theoretically answer questions about any corpus. ✅
- [ ] It is primarily parametric and requires a different model for each corpus.
- [ ] It is not suitable for fact-checking because of high hallucination occurrences.

#### Q18. Numerical values per input phrase for cohere.embed-english-light-v3.0?
- [ ] 256
- [x] 384 ✅
- [ ] 512
- [ ] 1024

#### Q19. What advantage does fine-tuning offer for improving model efficiency?
- [ ] It eliminates the need for annotated data during training.
- [ ] It reduces the number of tokens needed for model performance.
- [ ] It increases the model's context window size.
- [x] It improves the model's understanding of human preferences. ✅

#### Q20. Which settings are most likely to induce hallucinations and factual errors?
- [ ] `temperature = 0.5`, `top_p = 0.9`, `frequency_penalty = 0.5`
- [ ] `temperature = 0.0`, `top_p = 0.7`, `frequency_penalty = 1.0`
- [ ] `temperature = 0.2`, `top_p = 0.6`, `frequency_penalty = 0.8`
- [x] `temperature = 0.9`, `top_p = 0.8`, `frequency_penalty = 0.1` ✅

#### Q21. What happens to an endpoint's status after initiating a move to a different compartment?
- [ ] The endpoint becomes Inactive permanently, and you need to create a new endpoint.
- [ ] The endpoint is deleted and recreated in the new compartment.
- [ ] The status remains Active throughout the move.
- [x] The status changes to Updating during the move and returns to Active after completion. ✅

#### Q22. In Oracle Database 23ai, which data type should store image embeddings?
- [ ] Float32
- [ ] INT
- [ ] Double
- [x] VECTOR ✅

#### Q23. On-demand inferencing billing for a 200-char prompt + 500-char response?
- [x] 1 transaction per API call, regardless of length. ✅
- [ ] 200 transactions
- [ ] 700 transactions

#### Q24. Which of these does NOT apply when preparing PDFs for OCI Generative AI Agents?
- [x] Charts must be two-dimensional with labeled axes. ✅
- [ ] PDF files can include images and charts.
- [ ] Hyperlinks in PDFs are excluded from chat responses.
- [ ] Reference tables must be formatted with rows and columns.

#### Q25. Which parameter should be modified to ensure identical outputs for the same input?
- [ ] frequency_penalty
- [ ] temperature
- [x] seed ✅
- [ ] top_p

#### Q26. What happens when this line is executed?
`embed_text_response = generative_ai_inference_client.embed_text(embed_text_detail)`
- [x] It sends a request to the OCI Generative AI service to generate an embedding for the input text. ✅
- [ ] It initiates a connection to OCI and authenticates using the user's credentials.
- [ ] It processes and configures the OCI profile settings for the inference session.
- [ ] It initializes a pretrained OCI Generative AI model for use in the session.

#### Q27. Which phase of the RAG pipeline includes loading, splitting, and embedding documents?
- [ ] Retrieval
- [x] Ingestion ✅
- [ ] Generation
- [ ] Evaluation

#### Q28. In RAG, how might Groundedness differ from Answer Relevance?
- [x] Groundedness pertains to factual correctness, while Answer Relevance concerns query relevance. ✅
- [ ] Groundedness measures relevance to the user query, while Answer Relevance evaluates data integrity.
- [ ] Groundedness refers to contextual alignment, while Answer Relevance deals with syntactic accuracy.
- [ ] Groundedness focuses on data integrity, while Answer Relevance emphasizes lexical diversity.

#### Q29. In generating text with an LLM, what does greedy decoding entail?
- [ ] Picking a word based on its position in a sentence structure.
- [ ] Using a weighted random selection based on a modulated distribution.
- [x] Choosing the word with the highest probability at each step of decoding. ✅
- [ ] Selecting a random word from the entire vocabulary at each step.

#### Q30. What is the role of OnDemandServingMode in this snippet?
`chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id="ocidl.generativeaimodel.ocl.eu-frankfurt-1.XXXXXXXXXXXXXXXXXXXXXX")`
- [ ] It defines the retry strategy for handling failures during model inference.
- [ ] It initializes the model with the default configuration profile for inference.
- [x] It specifies that the Generative AI model should serve requests only on demand, rather than continuously. ✅
- [ ] It configures the model to use batch processing for requests.

#### Q31. Which best describes T-Few fine-tuning for LLMs?
- [x] It selectively updates only a fraction of the model's weights. ✅
- [ ] It increases the training time as compared to Vanilla fine-tuning.
- [ ] It does not update any weights but restructures the model architecture.
- [ ] It updates all the weights of the model uniformly.

#### Q32. How can you affect the probability distribution over the vocabulary of an LLM?
- [ ] By adjusting the token size during the training phase.
- [ ] By modifying the model's training data.
- [x] By using techniques like prompting and training. ✅
- [ ] By restricting the vocabulary used in the model.

#### Q33. Which of these is NOT a supported knowledge base data type for OCI Generative AI Agents?
- [ ] OCI Object Storage files with text and PDFs
- [ ] OCI Search with OpenSearch
- [ ] Oracle Database 23ai vector search
- [x] Custom-built file systems ✅

#### Q34. What happens to chat data and retrieved context after the session ends?
- [ ] They are archived for audit purposes.
- [ ] They are stored for training the Large Language Models (LLMs).
- [x] They are permanently deleted and not retained. ✅
- [ ] They are stored in isolation for future customer usage, ensuring maximum security but not used for training.

#### Q35. Why did the model claim smartwatch features not in documentation?
- [x] The model is hallucinating, confidently generating responses that are not grounded in factual or provided data. ✅
- [ ] The model encountered a prompt that was too ambiguous, leading to random outputs.
- [ ] The model is overfitting to specific details from unrelated training data, causing inaccuracies.
- [ ] The model was unable to access the company's database, so it guessed based on similar products.

#### Q36. How is `totalTrainingSteps` calculated during fine-tuning?
- [ ] `totalTrainingSteps = (totalTrainingEpochs * trainingBatchSize) / size(trainingDataset)`
- [ ] `totalTrainingSteps = (totalTrainingEpochs + size(trainingDataset)) * trainingBatchSize`
- [ ] `totalTrainingSteps = (size(trainingDataset) * trainingBatchSize) / totalTrainingEpochs`
- [x] `totalTrainingSteps = (totalTrainingEpochs * size(trainingDataset)) / trainingBatchSize` ✅

#### Q37. When specifying a data source, what does enabling multi-modal parsing do?
- [ ] Automatically tags files and folders in the bucket.
- [ ] Merges multiple data sources into a single knowledge base after parsing the files.
- [x] Parses and includes information from charts and graphs in the documents. ✅
- [ ] Parses and converts non-supported file formats into supported ones.

#### Q38. When does a chain typically interact with memory in a run within LangChain?
- [ ] Continuously throughout the entire chain execution process.
- [x] After user input but before chain execution, and again after core logic but before output. ✅
- [ ] Before user input and after chain execution.
- [ ] Only after the output has been generated.

#### Q39. What does a cosine distance of 0 indicate about two embeddings?
- [x] They are similar in direction. ✅
- [ ] They are unrelated.
- [ ] They have the same magnitude.
- [ ] They are completely dissimilar.

#### Q40. In which scenario is soft prompting more appropriate?
- [ ] When the model requires continued pretraining on unlabeled data.
- [x] When there is a need to add learnable parameters to a LLM without task-specific training. ✅
- [ ] When there is a significant amount of labeled, task-specific data available.
- [ ] When adapting to a domain not originally trained on.

#### Q41. What should a company do if its OCI Generative AI model is deprecated?
- [ ] Ignore the notification; deprecated models remain available indefinitely.
- [x] Continue using it for now but plan to migrate before retirement. ✅
- [ ] Request an extension to continue using the model after it is retired.
- [ ] Immediately stop and switch to a newer model.

#### Q42. Which fine-tuning methods are supported by the cohere.command-r-08-2024 model?
- [ ] T-Few and Vanilla
- [ ] LORA and Vanilla
- [x] T-Few and LORA ✅
- [ ] T-Few, LoRA, and Vanilla

#### Q43. Approach to declaratively connect/replace components in an LLM app?
- [ ] Use agents.
- [x] Use LangChain Expression Language (LCEL). ✅
- [ ] Use Python classes like LLMChain.
- [ ] Use prompts.

#### Q44. What does the OCI Generative AI service offer to users?
- [ ] Only pretrained LLMs with customization options.
- [ ] A service requiring users to share GPUs for deploying LLMs.
- [ ] A limited platform that supports chat-based LLMs without hosting capabilities.
- [x] Fully managed LLMs along with the ability to create custom fine-tuned models. ✅

#### Q45. When is fine-tuning an appropriate method for customizing an LLM?
- [x] When the LLM does not perform well on a particular task and the data required to adapt the LLM is too large for prompt engineering. ✅
- [ ] When the LLM already understands the topics necessary for text generation.
- [ ] When you want to optimize the model without any instructions.
- [ ] When the LLM requires access to the latest data for generating outputs.

#### Q48. What is the purpose of this `endpoint` variable?
endpoint = "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com"
- [ ] It sets the retry strategy for the inference client.
- [ ] It specifies the availability domain where the model is hosted.
- [ ] It stores the OCI API key required for authentication.
- [x] It defines the URL of the OCI Generative AI inference service. ✅

#### Q49. Which statement regarding fine-tuning and PEFT is correct?
- [x] Fine-tuning trains the entire model (costly); PEFT updates only a small subset of parameters, minimizing compute and data needs. ✅
- [ ] PEFT replaces the entire model architecture and is more data-intensive than fine-tuning.
- [ ] Both require training from scratch and are equally intensive.
- [ ] Neither modifies the model; they only differ in data type used.

#### Q50. When activating content moderation in OCI Generative AI Agents, what can you specify?
- [x] Whether moderation applies to user prompts, generated responses, or both. ✅
- [ ] The maximum file size for input data.
- [ ] The threshold for language complexity in responses.
- [ ] The type of vector search used for retrieval.
