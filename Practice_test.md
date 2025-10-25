# Practice Exam: OCI Data Science Professional Certification (1Z0-1110-25)

#### Q1. A company wants to integrate an LLM into its customer support chatbot using OCI. What is the fastest way to deploy and test the model?
- [x] Using AI Quick Actions to quickly deploy a pretrained LLM ✅
- [ ] Building a deep learning model in Jupyter Notebook
- [ ] Training a custom model from scratch using OCI AutoML
- [ ] Manually configuring a model deployment using OCI SDK

#### Q2. A healthcare company needs to redact personal details (such as names, emails, and phone numbers) from patient records before sharing them with a research institute. Which operator is best suited for this task?
- [ ] Anomaly Detection Operator
- [ ] Clustering Operator
- [x] PII Detection Operator ✅
- [ ] Forecasting Operator

#### Q3. Which correlation method is used to measure the relationship between two categorical variables in ADS?
- [ ] Chi-square test
- [x] Cramer's V method ✅
- [ ] Spearman correlation coefficient
- [ ] Pearson correlation coefficient

#### Q4. When deploying an RAG application to OCI Data Science, what is the correct sequence of steps you would need to follow?
- [x] Load documents.  
Split documents.  
Embed documents.  
Create vector database from documents.  
Create retriever.  
Create chain.  
Create model.  
Prepare model artifacts.  
Verify model.  
Save model.  
Deploy model. ✅
- [ ] Create model.  
Load documents.  
Split documents.  
Embed documents.  
Create vector database from documents.  
Create retriever.  
Create chain.  
Prepare model artifacts.  
Verify model.  
Save model.  
Deploy model.

#### Q5. Once you deploy the LLM using AI Quick Actions, how can you invoke your model?
- [ ] Through API
- [ ] Through CLI
- [ ] Through only CLI
- [x] Through API and CLI ✅

#### Q6. As a data scientist for a hardware company, you have been asked to predict the revenue demand for the upcoming quarter. You develop a time series forecasting model to analyze the data.  
Which is the correct sequence of steps to predict the revenue demand values for the upcoming quarter?
- [ ] Prepare model, deploy, verify, save, predict.
- [ ] Verify, prepare model, deploy, save.
- [ ] Predict, deploy, save, verify, prepare model.
- [x] Prepare model, verify, save, deploy, predict. ✅

#### Q7. A bike sharing platform has collected user commute data for the past three years. For increasing profitability and making useful inferences, a machine learning model needs to be built from the accumulated data.  
Which option has the correct order of the required machine learning tasks for building a model?
- [x] Data Access, Data Exploration, Feature Exploration, Feature Engineering, Modeling ✅
- [ ] Data Access, Feature Exploration, Data Exploration, Feature Engineering, Modeling
- [ ] Data Access, Data Exploration, Feature Engineering, Feature Exploration, Modeling
- [ ] Data Access, Feature Exploration, Feature Engineering, Data Exploration, Modeling

#### Q8. A company is running a job in OCI Data Science Jobs and wants to ensure that the infrastructure is deprovisioned immediately after the job completes to avoid unnecessary costs.  
What happens when the job ends?
- [x] The infrastructure is automatically deprovisioned. ✅
- [ ] The infrastructure remains active for 30 days.
- [ ] The job artifact is deleted.
- [ ] The compute shape is reset to default.

#### Q9. You are running a pipeline in OCI Data Science Service and want to override some of the pipeline's default settings.  
Which statement is true about overriding pipeline defaults?
- [ ] Pipeline defaults cannot be overridden once the pipeline has been created.
- [ ] Pipeline defaults can be overridden only by an administrator.
- [x] Pipeline defaults can be overridden only during pipeline creation. ✅
- [ ] Pipeline defaults can be overridden before starting the pipeline execution.

#### Q10. Which is NOT a supported encryption algorithm in OCI Vault?
- [ ] AES (Advanced Encryption Standard)
- [x] SHA-256 (Secure Hash Algorithm 256-bit) ✅
- [ ] ECDSA (Elliptic Curve Digital Signature Algorithm)
- [ ] RSA (Rivest-Shamir-Adleman)

#### Q11. A data scientist needs to securely access an external database from their notebook session. What is the best way to store the credentials?
- [x] Save the credentials in OCI Vault and retrieve them programmatically when needed. ✅
- [ ] Store the credentials in a plaintext configuration file.
- [ ] Hardcode the credentials in the Jupyter Notebook.
- [ ] Share the credentials via email with team members.

#### Q12. A team wants to create a sophisticated autoscaling query that combines multiple metrics using logical operators. Which option should they use?
- [ ] Predefined metrics
- [x] Custom scaling metric with NQL expressions ✅
- [ ] Load balancer scaling
- [ ] Cooldown periods

#### Q13. What model parameter value are you most likely to use if you are not sure of your selection while configuring the Forecasting operator?
- [ ] arima
- [ ] autotts
- [ ] prophet
- [x] auto ✅

#### Q14. A data scientist is working on a fraud detection model. They need to store the trained model so that it can be versioned, tracked, and later deployed without modification. Which feature should they use?
- [ ] Model Explainability
- [x] Model Catalog ✅
- [ ] Model Deployment
- [ ] Hyperparameter Tuning

#### Q15. What happens when a model deployment in OCI Data Science is deactivated?
- [ ] The model deployment metadata is erased along with the model artifacts.
- [ ] The model remains active but stops accepting new inference requests.
- [ ] The deployed model is permanently deleted, and predictions are no longer possible.
- [x] The model's HTTP endpoint becomes unavailable, but metadata is preserved. ✅

#### Q16. You have custom data and you want to customize an off-the-shelf LLM and deploy it quickly. How can AI Quick Actions help you?
- [ ] To pretrain the LLM
- [ ] To deploy the off-the-shelf model
- [x] To fine-tune the model and deploy ✅

#### Q17. You are a data scientist working on a census dataset. You have decided to use Oracle AutoML Pipeline for automating your machine learning task and want to ensure that two of the features ("Age" and "Education") are part of the final model that the AutoML creates.  
To ensure these features are not dropped during the feature selection phase, what would be the best way to define the `min_features` argument in your code?
- [x] `min_features = ['Age', 'Education']` ✅
- [ ] `0 < min_features <= 0.9`
- [ ] `min_features = 'Age' && min_features = 'Education'`
- [ ] `0 < min_features <= 2`

#### Q18. A team wants to use CPU utilization as a metric to trigger autoscaling. Which type of autoscaling policy should they configure?
- [ ] Load balancer scaling
- [ ] Manual scaling
- [ ] Custom scaling metric
- [x] Predefined metric ✅

#### Q19. Which statement is true regarding autoscaling configuration for an existing model deployment in an Active state in Oracle Data Science?
- [ ] Only non-infrastructure-related aspects can be modified for an active model deployment.
- [x] Changes to the Autoscaling Scaling Policy fields must occur one field at a time, without simultaneous changes to other configurations. ✅
- [ ] You must disable the model deployment to update the Autoscaling Scaling Policy fields.
- [ ] You can modify the Autoscaling Scaling Policy fields and other configurations simultaneously.

#### Q20. What is the purpose of a dynamic group in OCI?
- [ ] To define storage limits for data science resources
- [x] To manage API access for resources such as notebook sessions ✅
- [ ] To allocate computing resources dynamically
- [ ] To group individual users for easier management

#### Q21. What triggers the automation of the MLOps pipeline?
- [ ] Manual intervention by data scientists
- [ ] User feedback
- [ ] Random system updates
- [x] Changes in data, monitoring events, or calendar intervals ✅

#### Q22. A data scientist is running a long-term experiment in an OCI notebook session. They need to save results even if they deactivate the session to reduce costs.  
What should they do?
- [ ] Save results only in the boot volume, as it is retained indefinitely.
- [ ] Use default networking to automatically back up results to OCI Object Storage.
- [x] Store all results in the block storage, as it persists after deactivation. ✅
- [ ] Keep the session active indefinitely to prevent data loss.

#### Q23. Which statement is incorrect regarding the benefits of autoscaling for model deployment in Oracle Data Science?
- [ ] Users can set customizable triggers for autoscaling using MQL expressions to tailor the scaling behavior according to specific needs.
- [ ] Autoscaling, in conjunction with load balancers, enhances availability by rerouting traffic to healthy instances in case of instance failure.
- [ ] Autoscaling dynamically adjusts compute resources based on real-time demand, ensuring efficient handling of varying loads.
- [x] By using autoscaling, the cost of deployment remains constant irrespective of resource utilization. ✅

#### Q24. Where are the training job outputs stored after fine-tuning is completed?
- [ ] In the local storage of the training instance
- [x] In an OCI Object Storage bucket ✅
- [ ] In a temporary cache that is cleared after job completion
- [ ] Directly in the OCI Model Catalog

#### Q25. How can a team ensure that data processing occurs before model training in a pipeline?
- [ ] By using the same programming language for all steps
- [ ] By overriding the default configuration
- [x] By setting dependencies between steps ✅
- [ ] By increasing the block volume size

#### Q26. Which statement best describes Oracle Cloud Infrastructure Data Science Jobs?
- [ ] Jobs lets you define and run all Oracle Cloud DevOps workloads.
- [x] Jobs lets you define and run repeatable tasks on fully-managed infrastructure. ✅
- [ ] Jobs lets you define and run repeatable tasks on fully-managed, third-party cloud infrastructure.
- [ ] Jobs lets you define and run repeatable tasks on customer-managed infrastructure.

#### Q27. What is the key difference between PDP (Partial Dependence Plot) and ICE (Individual Conditional Expectation) in ADS?
- [ ] PDP is used for classification, while ICE is only used for regression.
- [ ] PDP is a supervised learning technique, while ICE is used for unsupervised learning.
- [ ] PDP works only for categorical features, while ICE works only for continuous features.
- [x] PDP provides feature-level insights, while ICE provides sample-level insights. ✅

#### Q28. You are a data scientist using Oracle AutoML to produce a model and you are evaluating the score metric for the model.  
Which two prevailing metrics would you use for evaluating the multiclass classification model?
- [x] Recall ✅
- [x] F1 score ✅
- [ ] Mean squared error
- [ ] R-squared
- [ ] Explained variance score

#### Q29. A data scientist is using an AI model to predict fraudulent transactions. A financial regulator asks why a specific transaction was flagged as fraud.  
Which technique should the data scientist use?
- [ ] Global Explanation
- [x] Local Explanation ✅
- [ ] What-If Explanation
- [ ] Feature Permutation Importance

#### Q30. While working with Git on Oracle Cloud Infrastructure (OCI) Data Science, you notice that two of the operations are taking more time than the others due to your slow internet speed.  
Which two operations would experience the delay?
- [ ] Moving changes into the staging area for the next commit
- [ ] Converting an existing local project folder to Git repository
- [ ] Making a commit that is taking a snapshot of the local repository for the next push
- [x] Pushing changes to a remote repository ✅
- [x] Updating the local repo to match the content from a remote repository ✅

#### Q31. You are given data with sensitive information like emails and contact numbers of the customer. You need to redact this information before using it further.  
What OCI Data Science Operator can you use for this purpose?
- [x] PII ✅
- [ ] Anomaly
- [ ] Forecasting

#### Q32. You want to create a user group for a team of external data science consultants. The consultants should only have the ability to view data science resource details but not the ability to create, delete, or update data science resources.  
What verb should you write in the policy?
- [ ] Inspect
- [x] Read ✅
- [ ] Use
- [ ] Manage

#### Q33. What is the primary reason for performing feature scaling in machine learning models?
- [ ] To convert categorical data into numerical form
- [x] To bring features on to the same scale ✅
- [ ] To make the dataset smaller for faster computation
- [ ] To automatically detect missing values and fill them with mean or median

#### Q34. What is the difference between a job and a job run in OCI Data Science Jobs?
- [ ] A job is used for model training, while a job run is used for batch inference.
- [x] A job is a template, while a job run is a single execution of that template. ✅
- [ ] A job is a single execution, while a job run is a template.
- [ ] A job is immutable, while a job run can be modified.

#### Q35. What is the primary goal of the loss function in model training?
- [ ] To determine the best algorithm for training the model
- [ ] To maximize the likelihood of data points fitting the model
- [x] To compare predicted values with true target values and quantify their difference ✅
- [ ] To update the model parameters to optimize performance

#### Q36. A company has trained a machine learning model and wants to fine-tune it by experimenting with hyperparameter values based on prior experience.  
What approach should they take?
- [ ] Apply the detailed search space for broader tuning.
- [ ] Skip hyperparameter tuning altogether.
- [x] Define a custom search space with specific hyperparameter values. ✅
- [ ] Use the built-in perfunctory search strategy.

#### Q37. A team of data scientists is working on multiple machine learning models for fraud detection. They want to collaborate in a structured manner.  
What option is available to create a Data Science Project in OCI?
- [ ] Can be created only through the OCI Console UI
- [x] Can be created through either the OCI Console UI or the ADS SDK ✅
- [ ] Can be created using a command-line interface (CLI) only
- [ ] Can be created only through the ADS SDK

#### Q39. A data scientist wants to develop a PySpark application iteratively using a sample of their dataset.  
Which environment is recommended for this purpose?
- [ ] OCI Compute
- [ ] OCI Object Storage
- [ ] OCI Virtual Cloud Network
- [x] OCI Data Science notebook session ✅

#### Q40. What is the final step after running the Oracle Resource Manager stack for Data Science configuration?
- [ ] Modifying the Terraform script in GitHub
- [x] Adding users to the automatically created user group ✅
- [ ] Deleting the default compartment
- [ ] Creating an additional stack for security configuration

#### Q41. Which type of data is NOT available in Oracle Open Data?
- [ ] Protein sequences and genomic data
- [x] Financial transaction data ✅
- [ ] Geospatial data from satellite systems
- [ ] Annotated text files

#### Q42. A data scientist updates an IAM policy to grant their notebook session access to an Object Storage bucket. However, the notebook still cannot access the bucket.  
What is the likely reason?
- [x] The resource principal token is still cached. ✅
- [ ] Object Storage does not support access from notebooks.
- [ ] The user needs to restart the entire OCI environment.
- [ ] The IAM policy is incorrect.

#### Q43. Which resource types are included in the default matching rules of the Data Science Service template?
- [ ] datasciencenetwork, datasiencedatabase, datasciencebackup
- [x] datasciencemodeldeployment, datasciencenotebooksession, datasciencejobrun ✅
- [ ] datascienceanalytics, datasiencemonitoring, datasciencebatchjob
- [ ] datascienceobjectstorage, datasciencecomputeinstance, datasciencemodeltraining

#### Q44. A data scientist is working on a project to train a machine learning model to identify tigers in images.  
What is the first step they need to take before training the model?
- [ ] Deploy the model.
- [ ] Use OCI Vision Services.
- [x] Label the images with “tiger” or “not tiger”. ✅
- [ ] Analyze customer feedback.

#### Q45. Which two statements are true about Oracle Cloud Infrastructure (OCI) Open Data Service?
- [ ] A primary goal of Open Data is for users to contribute to the data repositories in order to expand the content offered.
- [ ] Subscribers can pay and log into Open Data to view curated datasets that are otherwise not available to the public.
- [ ] Open Data includes text and image data repositories for AI and ML. Audio and video formats are not available.
- [x] Open Data is a dataset repository made for the people that create, use, and manipulate datasets. ✅
- [x] Each dataset in Open Data consists of code and tooling usage examples for consumption and reproducibility. ✅

#### Q46. You have trained a binary classifier for a loan application and saved this model into the model catalog. A colleague wants to examine the model, and you need to share it.  
From the model catalog, which model artifacts can be shared?
- [ ] Models and metrics only
- [ ] Metadata, hyperparameters, and metrics only
- [x] Models, model metadata, hyperparameters, and metrics ✅
- [ ] Model metadata and hyperparameters only

#### Q47. Arrange the steps of Git operations in correct order:
1. Install, configure, and authenticate Git.  
2. Configure SSH keys for the Git repository.  
3. Create a local and remote Git repository.  
4. Commit files to the local Git repo.  
5. Push the commit to the remote Git repo.
- [x] 2, 3, 1, 4, 5 ✅
- [ ] 4, 2, 3, 1, 5  
- [ ] 3, 5, 1, 2, 4  
- [ ] 1, 2, 3, 4, 5  

#### Q48. You have been given a collection of digital files for a business audit and want to annotate them using OCI Data Labeling.  
Which three types of files could this tool annotate?
- [x] Video footage of a conversation in a conference room ✅
- [ ] A type-written document that details an annual budget
- [x] Images of computer server racks ✅
- [ ] A collection of purchase orders for office supplies
- [x] An audio recording of a phone conversation ✅

#### Q49. You just started as a data scientist at a healthcare company and are analyzing a deep neural network model built using ECG records. There are no details about the framework used.  
What is the best way to find more details about the model inside the model catalog?
- [ ] Check for Provenance details.
- [ ] Refer to the code inside the model.
- [x] Check for Model Taxonomy details. ✅
- [ ] Check for metadata tags.

#### Q50. A user wants to fetch data from an Autonomous Database in OCI without using a database wallet.  
What must they do?
- [ ] Enable API authentication in the database console.  
- [ ] Use `ads.read_sql` without any additional parameters.  
- [x] Provide the hostname and port number in the `connection_parameters` dictionary. ✅  
- [ ] Use an HTTP request to retrieve database records.




