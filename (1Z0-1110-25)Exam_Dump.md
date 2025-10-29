# Oracle Exam 1Z0-1110-25 : Oracle Cloud Infrastructure 2025 Data Science Professional

#### Q1. You are setting up a fine-tuning job for a pre-trained model on Oracle Data Science. You obtain the pre-trained model from HuggingFace, define the training job using the ADS Python API, and specify the OCI bucket. The training script includes downloading the model and dataset.  
Which of the following steps will be handled automatically by the ADS during the job run?
- [ ] Fetching the source code from GitHub and checking out the specific commit.
- [ ] Setting up the conda environment and installing additional dependencies.
- [ ] Specifying the replica and shape of instances required for the training job.
- [x] Saving the outputs to OCI Object Storage once the training finishes. ✅

#### Q2. You want to use ADSTuner to tune the hyperparameters of a supported model you recently trained. You have just started your search and want to reduce the computational cost as well as assess the quality of the model class that you are using.  
What is the most appropriate search space strategy to choose?
- [ ] ADSTuner doesn’t need a search space to tune the hyperparameters.
- [ ] Pass a dictionary that defines a search space.
- [x] Perfunctory. ✅
- [ ] Detailed.

#### Q3. You are a data scientist designing an air traffic control model, and you choose to leverage Oracle AutoML. You understand that the Oracle AutoML pipeline consists of multiple stages and automatically operates in a certain sequence.  
What is the correct sequence for the Oracle AutoML pipeline?
- [ ] Algorithm selection, Adaptive sampling, Feature selection, Hyperparameter tuning.
- [ ] Adaptive sampling, Feature selection, Algorithm selection, Hyperparameter tuning.
- [x] Adaptive sampling, Algorithm selection, Feature selection, Hyperparameter tuning. ✅
- [ ] Algorithm selection, Feature selection, Adaptive sampling, Hyperparameter tuning.

#### Q4. Which approach does Oracle AutoML use to avoid the cold start problem?
- [ ] Genetic evolutionary algorithms to evolve new models dynamically.
- [ ] Exhaustive grid search to evaluate every possible model configuration.
- [x] Meta-learning to predict algorithm performance on unseen data sets. ✅
- [ ] Randomized hyperparameter tuning to generate diverse models.

#### Q5. You are working as a data scientist for a healthcare company. They decide to analyze the data to find patterns in a large volume of electronic medical records. You are asked to build a PySpark solution to analyze these records in a JupyterLab notebook.  
What is the order of recommended steps to develop a PySpark application in Oracle Cloud Infrastructure (OCI) Data Science?  
- [ ] Install a Spark conda environment. Configure core-site.xml. Launch a notebook session. Create a Data Flow application with the Accelerated Data Science (ADS) SDK. Develop your PySpark application.  
- [x] Launch a notebook session. Install a PySpark conda environment. Configure core-site.xml. Develop your PySpark application. Create a Data Flow application with the Accelerated Data Science (ADS) SDK. ✅  
- [ ] Launch a notebook session. Configure core-site.xml. Install a PySpark conda environment. Develop your PySpark application. Create a Data Flow application with the Accelerated Data Science (ADS) SDK.  
- [ ] Configure core-site.xml. Install a PySpark conda environment. Create a Data Flow application with the Accelerated Data Science (ADS) SDK. Develop your PySpark application. Launch a notebook session.

#### Q6. As a data scientist, you are working on a movie-recommendation application where you have a very large movie dataset. Which Oracle Cloud Infrastructure (OCI) services should you use to develop interactive Spark applications and deploy Spark workloads?  
- [ ] Data Science and Vault  
- [x] Data Flow and Data Science ✅  
- [ ] Analytics Cloud and Data Flow  
- [ ] Data Integration and Vault

#### Q7. A data scientist is training a regression model and notices that the predicted values significantly deviate from the actual target values. What should they examine first?  
- [ ] The open-source frameworks being used in the project  
- [ ] The score function to determine the best algorithm  
- [ ] The update function to see if model parameters are changing  
- [x] The loss function to measure the error and guide optimization. ✅

#### Q8. When preparing your model artifact to save it to the Oracle Cloud Infrastructure (OCI) Data Science model catalog, you create a score.py file. What is the purpose of the score.py file?  
- [x] Execute the inference logic code. ✅  
- [ ] Define the compute scaling strategy.  
- [ ] Configure the deployment infrastructure.  
- [ ] Define the inference server dependencies.

#### Q9. You realize that your model deployment is about to reach its utilization limit. What would you do to avoid the issue before requests start to fail?  
- [x] Update the deployment to add more instances. ✅  
- [ ] Update the deployment to use a larger virtual machine (more CPUs/memory).  
- [ ] Delete the deployment.  
- [ ] Update the deployment to use fewer instances.  
- [ ] Reduce the load balancer bandwidth limit so that fewer requests come in.

#### Q10. You need to build a machine learning workflow that has sequential and parallel steps. You have decided to use the Oracle Cloud Infrastructure (OCI) Data Science Pipeline feature. How is Directed Acyclic Graph (DAG) having sequential and parallel steps built using Pipeline?  
- [ ] Using dependencies  
- [ ] Using environment variables  
- [x] By running a Pipeline ✅  
- [ ] Using Pipeline Designer

#### Q11. For your next data science project, you need access to public geospatial images.  
Which Oracle Cloud service provides free access to those images?  
- [ ] Oracle Analytics Cloud  
- [ ] Oracle Cloud Infrastructure Data Science  
- [x] Oracle Open Data ✅  
- [ ] Oracle Big Data Service

#### Q12. As a data scientist, you are working on a global health data set that has data from more than 50 countries. You want to encode three features such as 'countries', 'race' and 'body organ' as categories.  
Which option would you use to encode the categorical feature?  
- [ ] auto_transform()  
- [ ] show_in_notebook()  
- [x] OneHotEncoder() ✅  
- [ ] DataFrameLabelEncoder()

#### Q13. While reviewing your data, you discover that your data set has a class imbalance. You are aware that the Accelerated Data Science (ADS) SDK provides multiple built-in automatic transformation tools for data set transformation.  
Which would be the right tool to correct any imbalance between the classes?  
- [ ] auto_transform()  
- [ ] visualize_transforms()  
- [x] sample() ✅  
- [ ] suggest_recommendations()

#### Q14. You are a data scientist trying to load data into your notebook session. You understand that the Accelerated Data Science (ADS) SDK supports loading various data formats.  
Which of the following THREE are ADS supported data formats?  
- [x] JSON ✅  
- [ ] DOCX  
- [x] Pandas DataFrame ✅  
- [ ] Raw Images  
- [x] XML ✅

#### Q15. A data scientist is evaluating a multiclass classification model and notices that the precision and recall values vary significantly across different class labels. What should they use to analyze the model’s performance in detail?  
- [ ] Mean Squared Error (MSE)  
- [x] Confusion Matrix and Precision-Recall by Label Charts ✅  
- [ ] ROC Curve  
- [ ] Residuals QQ Plot

#### Q16. You train a model to predict housing prices for your city. Which two metrics from the Accelerated Data Science (ADS) ADSEvaluator class can you use to evaluate the regression model?  
- [x] Explained Variance Score ✅  
- [x] Mean Absolute Error ✅  
- [ ] F-1 Score  
- [ ] Weighted Recall  
- [ ] Weighted Precision  

#### Q17. As a data scientist, you are tasked with creating a model training job that is expected to take different hyperparameter values on every run. What is the most efficient way to set those parameters with Oracle Data Science Jobs?  
- [ ] Create a new job every time you need to run your code and pass the parameters as environment variables.  
- [ ] Create a new job by setting the required parameters in your code, and create a new job for every code change.  
- [x] Create your code to expect different parameters either as environment variables or as command line arguments, which are set on every job run with different values. ✅  
- [ ] Create your code to expect different parameters as command line arguments, and create a new job every time you run the code.  

#### Q18. You have received machine learning model training code, without clear information about the optimal shape to run the training. How would you proceed to identify the optimal compute shape for your model training that provides a balanced cost and processing time?  
- [x] Start with a smaller shape and monitor the Job Run metrics and time required to complete the model training. If the compute shape is not fully utilized, tune the model parameters, and re-run the job. ✅  
- [ ] Start with a random compute shape and monitor the utilization metrics and time required to finish the model training. Perform model training optimisations and performance tests in advance to identify the right compute shape before running the model training as a job.  
- [ ] Start with the strongest compute shape Job’s support and monitor the Job Run metrics and time required to complete the model training. Tune the model so that it utilizes as much compute resources as possible, even at an increased cost.  
- [ ] Start with a smaller shape and monitor the utilization metrics and time required to complete the model training. If the compute shape is fully utilized, change to compute that has more resources and re-run the job.  

#### Q19. You are using jobs for model training, and you want to interrupt the model training and exit your job when a specific metric value has been reached. What would be the best way to do this with Data Science Jobs?  
- [ ] Jobs exit and de-provision the infrastructure when the code is done, by either naturally ending the code flow or triggering the `exit()` function.  
- [x] In your model training code, you should use the OCI SDK and call the job’s `cancel()` function to stop the training. ✅  
- [ ] You have to monitor the job process outside of the job run and cancel the job using the OCI SDK, once the metric value has been reached.  
- [ ] You cannot cancel and exit a job in Data Science Jobs before your code has finished executing.  

#### Q20. You want to install a list of Python packages on your data science notebook session while creating the instance. Which option will allow you to do the same?  
- [x] Using runtime configuration ✅  
- [ ] Invoking public endpoint  
- [ ] Using storage mounts  

#### Q21. A company wants to configure its OCI tenancy for Data Science quickly without manually setting up policies and groups. What is the best approach?
- [ ] Deploy a Kubernetes cluster with preconfigured access roles.  
- [x] Use the Data Science Service template in Oracle Resource Manager. ✅  
- [ ] Manually configure user groups, dynamic groups, and policies from the OCI Identity console.  
- [ ] Create a custom virtual network and security list before configuring policies.

#### Q22. You want to ensure that all stdout and stderr from your code are automatically collected and logged, without implementing additional logging in your code. How would you achieve this with Data Science Jobs?
- [ ] Create your own log group and use a third-party logging service to capture job run details for log collection and storing.  
- [ ] You can implement custom logging in your code by using the Data Science Jobs logging service.  
- [ ] Make sure that your code is using the standard logging library and then store all the logs to Object Storage at the end of the job.  
- [x] On job creation, enable logging and select a log group. Then, select either a log or the option to enable automatic log creation. ✅  

#### Q23. You want to build a multistep machine learning workflow by using the OCI Data Science Pipeline feature. How would you configure the conda environment to run a pipeline step?
- [x] Use environmental variables. ✅  
- [ ] Configure a compute shape.   
- [ ] Configure a block volume.  
- [ ] Use command-line variables.

#### Q24. During a job run, you receive an error message that no space is left on your disk device. To solve the problem, you must increase the size of the job storage. What would be the most efficient way to do this with Data Science Jobs?  
- [ ] Your code is using too much disk space. Refactor the code to identify the problem.  
- [ ] Create a new job with increased storage size and then run the job.  
- [x] Edit the job, change the size of the storage of your job, and start a new job run. ✅  
- [ ] On the job run, set the environment variable that helps increase the size of the storage.

#### Q25. A data scientist is working on a deep-learning project with TensorFlow and wants to ensure the same environment can be shared with colleagues. What is the best approach?  
- [x] Store the Conda environment as a published Conda environment in Object Storage. ✅  
- [ ] Create a new Conda environment every time a colleague needs access.  
- [ ] Manually install TensorFlow on each team member’s machine.  
- [ ] Copy and paste the package list into a text file for manual installation.

#### Q26. You have created a conda environment in your notebook session. This is the first time you are working with published conda environments. You have also created an Object Storage bucket with permission to manage the bucket.  
Which two commands are required to publish the conda environment?  
- [x] `odsc conda init --bucket_namespace <NAMESPACE> --bucket_name <BUCKET>` ✅  
- [x] `odsc conda publish --slug <SLUG>` ✅  
- [ ] `odsc conda list --override`  
- [ ] `odsc conda create --file manifest.yaml`  
- [ ] `conda activate /home/datascience/conda/<SLUG>/`  

#### Q27. You have just received a new dataset from a colleague. You want to quickly find out summary information about the dataset such as types of features, number of observations, and distributions of the data.  
Which Accelerated Data Science (ADS) SDK method from the `ADSDataset` class would you use?  
- [ ] `to_xgb()`  
- [ ] `show_corr()`  
- [x] `show_in_notebook()` ✅  
- [ ] `compute()`  


#### Q28. A data scientist is analyzing customer churn data and wants to visualize the relationship between monthly charges (continuous) and churn status (categorical).  
What is the best visualization that ADS will likely generate?  
- [ ] A line chart  
- [ ] A scatterplot  
- [x] A violin plot ✅  
- [ ] A histogram  

#### Q29. Once a LangChain application is deployed to OCI Data Science, what are two ways to invoke it as an endpoint?  
- [ ] Use `.predict` method or Use CLI  
- [x] Use `.invoke()` method or Use `.predict()` method ✅  
- [ ] Use CLI or Use `.invoke()`  


#### Q30. You need to make sure that the model you have deployed using AI Quick Actions is responding with suitable responses.  
How can AI Quick Actions help here?  
- [x] By evaluating the model ✅  
- [ ] By deploying the model  
- [ ] By fine-tuning the model  

#### Q31. You want to write a Python script to create a collection of different projects for your data science team.  
Which Oracle Cloud Infrastructure (OCI) Data Science interface would you use?  
- [ ] OCI Console  
- [ ] Mobile App  
- [ ] Command line interface (CLI)  
- [x] The OCI Software Development Kit (SDK) ✅  

#### Q32. Six months ago, you created and deployed a model that predicts customer churn for a call center. Initially, it was yielding quality predictions. However, over the last two months, users are questioning the credibility of the predictions.  
Which two methods would you employ to verify the accuracy of the model?  
- [ ] Retrain the model  
- [x] Validate the model using recent data ✅  
- [x] Drift monitoring ✅  
- [ ] Operational monitoring  
- [ ] Redeploy the model  

#### Q33. A data science team is using OCI Vault for storing secrets. They have rotated a secret’s contents, but their application still works without changes. Why?  
- [ ] The old version of the secret is still in use.  
- [ ] The application is caching the previous credentials.  
- [x] The secret’s OCID remains the same, allowing automatic updates. ✅  
- [ ] OCI Vault automatically updates code configurations.  


#### Q34. As a data scientist, you have stored sensitive data in a database. You need to protect this data by using a master encryption algorithm that uses symmetric keys.  
Which master encryption algorithm would you choose in the Oracle Cloud Infrastructure (OCI) Vault service?  
- [ ] Rivest–Shamir–Adleman Keys  
- [x] Advanced Encryption Standard (AES) Keys ✅  
- [ ] Triple Data Encryption Standard Algorithm  
- [ ] Elliptic Curve Cryptography Digital Signature Algorithm  


#### Q35. You are a data scientist with a set of text and image files that need annotation, and you want to use Oracle Cloud Infrastructure (OCI) Data Labeling.  
Which of the following **three annotation classes** are supported by the tool?  
- [ ] Key-Point and Landmark  
- [x] Named Entity Extraction ✅  
- [x] Object Detection ✅  
- [x] Classification (single/multi-label) ✅  
- [ ] Polygonal Segmentation  
- [ ] Semantic Segmentation

#### Q36. You are a data scientist and have a large number of legal documents that need to be classified. You decided to use OCI Data Labeling service to get your data labeled.  
What are the annotation classes available for annotating document data using OCI Data Labeling service?  
- [ ] Single, Multiple, Object Detection  
- [x] Single, Multiple, Entity Extraction ✅  
- [ ] Single, Multiple, Key Value  

#### Q37. You are building a machine learning model for predicting loan approvals, and the client wants to know which features are most influential in determining whether an applicant is approved. 
Which explainability method should you use?  
- [ ] What-If Explanation  
- [ ] ROC Curve  
- [x] Feature Permutation Importance ✅  
- [ ] Individual Conditional Expectation  

#### Q38. You want to evaluate the relationship between feature values and target variables. You have a large number of observations with near uniform distribution and highly correlated features.  
Which model explanation technique should you choose?  
- [ ] Local Interpretable Model-Agnostic Explanations (LIME)  
- [ ] Feature Permutation Importance Explanations  
- [ ] Feature Dependence Explanations  
- [x] Accumulated Local Effects ✅  

#### Q39. What is the sequence of steps you are likely to follow to use OCI Data Science Operator?  
- [x] 1. Install conda.  
  2. Initialize operator.  
  3. Configure operator.  
  4. Run operator.  
  5. Check results. ✅  
- [ ] Initialize operator → Install conda → Check results → Configure → Run  
- [ ] Initialize operator → Install conda → Check results → Configure → Run  
- [ ] Configure operator → Install conda → Initialize → Run → Check results

#### Q40. What detector in PII Operator are you likely to use if you need to obfuscate the detected sensitive information?  
- [x] Mask ✅  
- [ ] Remove  
- [ ] Anonymize  

#### Q41. You are creating an OCI Data Science job that runs on a recurring basis in a production environment, using sensitive data from Object Storage and saving the model to the model catalog.  
How would you design the authentication mechanism for the job?  
- [ ] Create a pre-authenticated request (PAR)  
- [ ] Package your personal OCI config file and keys  
- [ ] Store your personal OCI config in Vault  
- [x] Use the resource principal of the job run as the signer in the job code, ensuring there is a dynamic group for this job with access to Object Storage and the model catalog ✅  

#### Q42. As you are working in your notebook session, you find that your notebook session doesn’t have enough compute CPU and memory for your workload.  
How would you scale up your notebook session without losing your work?  
- [ ] Create a temporary bucket and copy manually  
- [ ] Download locally and reupload  
- [x] Ensure your files and environments are written to the block volume under `/home/datascience`, deactivate the notebook session, and activate a new one with a larger compute shape ✅  
- [ ] Deactivate session and re-create all files manually  

#### Q43. You are a data scientist working for a utilities company. You developed an anomaly detection algorithm and need to store the 2 GB model artifact into the model catalog.  
Which three interfaces could you use to save the model artifact?  
- [x] Oracle Cloud Infrastructure (OCI) Command Line Interface (CLI) ✅  
- [x] ODSC CLI ✅  
- [x] Accelerated Data Science (ADS) Software Development Kit (SDK) ✅  
- [ ] Git CLI  
- [ ] OCI Python SDK  

#### Q44. You developed a forecasting model that failed to run because not all third-party dependencies were included in the model artifact.  
What file should be modified to include the missing libraries?  
- [ ] score.py  
- [x] requirements.txt ✅  
- [ ] runtime.yaml  
- [ ] model_artifact_validate.py  

#### Q45. Which of the following statements is true regarding metric-based autoscaling in Oracle Data Science model deployment?  
- [ ] Only custom metrics can be used for metric-based autoscaling.  
- [ ] Multiple metric-based autoscaling policies can be added simultaneously.  
- [x] Metric-based autoscaling relies on performance metrics that are averaged across all instances in the model deployment resource. ✅  
- [ ] The cool-down period starts when the Model Deployment is first created.  

#### Q46. You are managing a machine learning model deployment in OCI Data Science. During peak hours, your application experiences spikes in traffic causing performance bottlenecks.  
Which configuration ensures cost efficiency while maintaining high performance?  
- [ ] Disabling the cool-down period  
- [ ] Predefining a fixed number of instances  
- [x] Setting the autoscaler with a wide scaling range (e.g., 2 to 15 instances) and enabling the load balancer to dynamically adjust bandwidth based on traffic ✅  
- [ ] Setting the autoscaler to have a narrow scaling range  

#### Q47. You used the `get_recommendations()` tool in ADS to see suggested transformations.  
Which option applies all the recommended transformations at once?  
- [x] auto_transform() ✅  
- [ ] fit_transform()  
- [ ] get_transformed_dataset()  
- [ ] visualize_transforms()  

#### Q48. After creating and opening a notebook session, from which two places can you access or install the ADS SDK?  
- [x] Conda environments in Oracle Cloud Infrastructure (OCI) Data Science ✅  
- [x] Python Package Index (PyPI) ✅  
- [ ] Oracle Machine Learning (OML)  
- [ ] Oracle Autonomous Data Warehouse  
- [ ] Oracle Big Data Service  

#### Q49. You are using a Git repository stored on GitHub to track notebooks.  
Which two statements are true?  
- [x] Once you have staged your changes, you run `git commit` to save a snapshot of your code. ✅  
- [ ] Only one of you has to clone the GitHub repo.  
- [ ] You should work on the same branch.  
- [ ] You do not have to clone the GitHub repo.  
- [x] To share your work, you commit and push it to GitHub. Your coworker can pull your changes into their notebook session. ✅  

#### Q50. Which two options signify the importance of having an MLOps strategy?  
- [x] Model decay – the data that was used to train the model is no longer relevant. ✅  
- [ ] Data needs preprocessing before training.  
- [ ] Data scientists need multiple tools for evaluation.  
- [x] Code updates to model training and deployment must be tested and deployed quickly. ✅  
- [ ] Data has to be secure both at rest and in transit.  



