# AWS Certified AI Practitioner (AIF-C01) Exam Cheat Sheet

## 📊 Exam Overview
- **Format**: 50 scored questions + 15 unscored
- **Duration**: Not specified in guide
- **Passing Score**: 700/1000
- **Question Types**: Multiple choice, Multiple response, Ordering, Matching, Case study

## 🎯 Domain Breakdown
- Domain 1: Fundamentals of AI and ML (20%)
- Domain 2: Fundamentals of Generative AI (24%)
- Domain 3: Applications of Foundation Models (28%)
- Domain 4: Guidelines for Responsible AI (14%)
- Domain 5: Security, Compliance, and Governance (14%)

---

## 🤖 Domain 1: Fundamentals of AI and ML (20%)

### Basic AI Terminology
- **AI**: Artificial Intelligence - broad field of creating intelligent machines
- **ML**: Machine Learning - subset of AI that learns from data
- **Deep Learning**: Subset of ML using neural networks with multiple layers
- **Neural Networks**: Computing systems inspired by biological neural networks
- **Computer Vision**: AI that interprets visual information
- **NLP**: Natural Language Processing - AI understanding human language
- **LLM**: Large Language Model - AI trained on vast text datasets
- **Algorithm**: Set of rules/instructions for solving problems
- **Training**: Process of teaching AI using data
- **Inference**: Using trained model to make predictions
- **Bias**: Systematic errors in AI predictions
- **Fairness**: Ensuring AI treats all groups equitably
- **Fit**: How well model matches training data

### Types of Learning
- **Supervised Learning**: Uses labeled data (input-output pairs)
  - Examples: Classification, Regression
- **Unsupervised Learning**: Finds patterns in unlabeled data
  - Examples: Clustering, Dimensionality reduction
- **Reinforcement Learning**: Learns through rewards and penalties

### Types of Data
- **Labeled vs Unlabeled**: Has target values vs doesn't
- **Structured**: Organized format (tables, databases)
- **Unstructured**: No predefined format (text, images, audio)
- **Tabular**: Rows and columns format
- **Time-series**: Data points over time
- **Image, Text**: Specific data modalities

### Types of Inference
- **Batch**: Processes multiple inputs at once
- **Real-time**: Processes single inputs immediately

### ML Development Lifecycle
1. **Data Collection**: Gathering relevant data
2. **EDA**: Exploratory Data Analysis
3. **Data Pre-processing**: Cleaning and preparing data
4. **Feature Engineering**: Creating relevant features
5. **Model Training**: Teaching the algorithm
6. **Hyperparameter Tuning**: Optimizing model settings
7. **Evaluation**: Testing model performance
8. **Deployment**: Making model available for use
9. **Monitoring**: Tracking model performance over time

### Model Performance Metrics
- **Accuracy**: Percentage of correct predictions
- **AUC**: Area Under ROC Curve
- **F1 Score**: Balance of precision and recall
- **Business Metrics**: ROI, cost per user, customer feedback

### Key AWS ML Services
- **Amazon SageMaker**: Full ML platform
- **Amazon Transcribe**: Speech-to-text
- **Amazon Translate**: Language translation
- **Amazon Comprehend**: Text analysis and NLP
- **Amazon Lex**: Conversational AI/chatbots
- **Amazon Polly**: Text-to-speech
- **SageMaker Data Wrangler**: Data preparation
- **SageMaker Feature Store**: Feature management
- **SageMaker Model Monitor**: Model monitoring

---

## 🎨 Domain 2: Fundamentals of Generative AI (24%)

### Core Generative AI Concepts
- **Tokens**: Basic units of text processing
- **Chunking**: Breaking text into smaller pieces
- **Embeddings**: Numerical representations of text/data
- **Vectors**: Mathematical representations in multi-dimensional space
- **Prompt Engineering**: Crafting effective inputs for AI models
- **Transformer-based LLMs**: Architecture using attention mechanisms
- **Foundation Models**: Large pre-trained models for multiple tasks
- **Multi-modal Models**: Handle multiple data types (text, image, audio)
- **Diffusion Models**: Generate data by reversing noise process

### Generative AI Use Cases
- Image, video, and audio generation
- Text summarization
- Chatbots and conversational AI
- Language translation
- Code generation
- Customer service agents
- Search enhancement
- Recommendation engines

### Foundation Model Lifecycle
1. **Data Selection**: Choosing training datasets
2. **Model Selection**: Picking appropriate architecture
3. **Pre-training**: Initial large-scale training
4. **Fine-tuning**: Specialized training for specific tasks
5. **Evaluation**: Testing model performance
6. **Deployment**: Making model available
7. **Feedback**: Continuous improvement

### Advantages of Generative AI
- **Adaptability**: Flexible for various tasks
- **Responsiveness**: Quick to new requirements
- **Simplicity**: Often easier than traditional approaches

### Limitations of Generative AI
- **Hallucinations**: Generating false information
- **Interpretability**: Difficult to understand decisions
- **Inaccuracy**: May produce incorrect results
- **Nondeterminism**: Same input may give different outputs

### Key AWS Generative AI Services
- **Amazon Bedrock**: Managed foundation models service
- **Amazon SageMaker JumpStart**: Pre-built ML solutions
- **PartyRock**: Amazon Bedrock Playground
- **Amazon Q**: AI assistant for AWS

### AWS Generative AI Benefits
- Lower barrier to entry
- Cost-effectiveness
- Speed to market
- Built-in security and compliance
- Scalable infrastructure

---

## 🏗️ Domain 3: Applications of Foundation Models (28%)

### Model Selection Criteria
- **Cost**: Pricing considerations
- **Modality**: Text, image, audio capabilities
- **Latency**: Response time requirements
- **Multi-lingual**: Language support
- **Model Size**: Computational requirements
- **Complexity**: Sophistication level
- **Customization**: Ability to modify
- **Input/Output Length**: Token limitations

### Inference Parameters
- **Temperature**: Controls randomness (0=deterministic, 1=creative)
- **Input/Output Length**: Token limits for requests and responses

### Retrieval Augmented Generation (RAG)
- **Definition**: Combining retrieval with generation
- **Purpose**: Provide external knowledge to models
- **Benefits**: Reduces hallucinations, adds current information
- **AWS Implementation**: Amazon Bedrock with knowledge bases

### Vector Database Services
- **Amazon OpenSearch Service**: Search and analytics
- **Amazon Aurora**: Relational database with vector support
- **Amazon Neptune**: Graph database
- **Amazon DocumentDB**: MongoDB-compatible
- **Amazon RDS for PostgreSQL**: With vector extensions

### Foundation Model Customization Approaches
- **Pre-training**: Training from scratch (most expensive)
- **Fine-tuning**: Adapting existing model (moderate cost)
- **In-context Learning**: Using examples in prompts (low cost)
- **RAG**: Adding external knowledge (moderate cost)

### Prompt Engineering Techniques
- **Zero-shot**: No examples provided
- **Single-shot**: One example provided
- **Few-shot**: Multiple examples provided
- **Chain-of-thought**: Step-by-step reasoning
- **Prompt Templates**: Reusable prompt structures
- **Negative Prompts**: What not to do
- **Context**: Background information
- **Instructions**: Specific directions

### Prompt Engineering Best Practices
- Be specific and concise
- Use multiple examples when helpful
- Implement guardrails
- Experiment with different approaches
- Provide clear context and instructions

### Prompt Engineering Risks
- **Exposure**: Revealing sensitive information
- **Poisoning**: Malicious training data
- **Hijacking**: Unauthorized control
- **Jailbreaking**: Bypassing safety measures

### Foundation Model Training Elements
- **Pre-training**: Initial training on large datasets
- **Fine-tuning**: Task-specific adaptation
- **Continuous Pre-training**: Ongoing training updates
- **Instruction Tuning**: Teaching to follow instructions
- **Transfer Learning**: Applying knowledge to new domains
- **RLHF**: Reinforcement Learning from Human Feedback

### Model Evaluation Methods
- **Human Evaluation**: People assess quality
- **Benchmark Datasets**: Standardized test sets
- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation
- **BLEU**: Bilingual Evaluation Understudy
- **BERTScore**: BERT-based evaluation metric

### Agents and Multi-step Tasks
- **Agents for Amazon Bedrock**: Automated multi-step task execution
- **Multi-step Tasks**: Complex workflows requiring multiple actions

---

## ⚖️ Domain 4: Guidelines for Responsible AI (14%)

### Features of Responsible AI
- **Bias**: Avoiding unfair discrimination
- **Fairness**: Equal treatment across groups
- **Inclusivity**: Representing diverse perspectives
- **Robustness**: Reliable performance
- **Safety**: Preventing harm
- **Veracity**: Truthfulness and accuracy

### Responsible AI Tools
- **Guardrails for Amazon Bedrock**: Safety controls
- **Amazon SageMaker Clarify**: Bias detection
- **SageMaker Model Monitor**: Performance tracking
- **Amazon A2I**: Human review workflows

### Legal Risks
- Intellectual property infringement
- Biased model outputs
- Loss of customer trust
- End user risks
- Hallucinations leading to harm

### Dataset Characteristics
- **Inclusivity**: Representing all relevant groups
- **Diversity**: Varied examples and perspectives
- **Curated Sources**: High-quality, verified data
- **Balanced**: Equal representation across categories

### Bias and Variance Effects
- Impact on demographic groups
- Inaccuracy in predictions
- Overfitting (too specific to training data)
- Underfitting (too general)

### Model Transparency vs Explainability
- **Transparent**: Can see how model works
- **Explainable**: Can understand model decisions
- **Trade-offs**: Complex models often less interpretable
- **Tools**: SageMaker Model Cards, open source models

### Environmental Considerations
- Energy consumption of training
- Carbon footprint of large models
- Sustainable AI practices

---

## 🔒 Domain 5: Security, Compliance, and Governance (14%)

### AWS Security Services for AI
- **IAM**: Identity and Access Management
- **Encryption**: At rest and in transit
- **Amazon Macie**: Data security and privacy
- **AWS PrivateLink**: Private connectivity
- **AWS Shared Responsibility Model**: Security division

### Data Security Best Practices
- **Data Quality Assessment**: Ensuring clean, accurate data
- **Privacy-Enhancing Technologies**: Protecting sensitive information
- **Data Access Control**: Limiting who can access data
- **Data Integrity**: Maintaining data accuracy and consistency
- **Data Lineage**: Tracking data origins and transformations
- **Data Cataloging**: Organizing and documenting data

### Security Considerations
- **Application Security**: Protecting AI applications
- **Threat Detection**: Identifying security risks
- **Vulnerability Management**: Addressing security weaknesses
- **Infrastructure Protection**: Securing underlying systems
- **Prompt Injection**: Preventing malicious prompts

### Compliance Standards
- **ISO**: International Organization for Standardization
- **SOC**: System and Organization Controls
- **Algorithm Accountability Laws**: Legal requirements for AI

### AWS Governance Services
- **AWS Config**: Configuration compliance
- **Amazon Inspector**: Security assessments
- **AWS Audit Manager**: Compliance auditing
- **AWS Artifact**: Compliance documentation
- **AWS CloudTrail**: Activity logging
- **AWS Trusted Advisor**: Best practice recommendations

### Data Governance Strategies
- **Data Lifecycles**: Managing data from creation to deletion
- **Logging**: Recording data activities
- **Residency**: Data location requirements
- **Monitoring**: Tracking data usage
- **Retention**: How long to keep data

### Governance Processes
- Policy development and maintenance
- Regular review schedules
- Review strategies and frameworks
- Generative AI Security Scoping Matrix
- Transparency standards
- Team training requirements

---

## 🛠️ Key AWS Services Summary

### Core AI/ML Services (Must Know)
- **Amazon Bedrock**: Foundation models service
- **Amazon SageMaker**: Complete ML platform
- **Amazon Q**: AI assistant
- **Amazon Comprehend**: NLP service
- **Amazon Lex**: Conversational AI
- **Amazon Polly**: Text-to-speech
- **Amazon Transcribe**: Speech-to-text
- **Amazon Translate**: Language translation
- **Amazon Rekognition**: Computer vision

### Supporting Services
- **Amazon S3**: Storage for data and models
- **Amazon EC2**: Compute for training and inference
- **AWS Lambda**: Serverless compute
- **Amazon OpenSearch**: Search and vector database
- **Amazon RDS**: Relational databases
- **Amazon DynamoDB**: NoSQL database

### Security & Governance
- **AWS IAM**: Identity and access management
- **AWS KMS**: Key management
- **Amazon Macie**: Data security
- **AWS CloudTrail**: Audit logging
- **AWS Config**: Configuration compliance

---

## 📝 Study Tips

### What's NOT in Scope (Don't Study These)
- Developing or coding AI/ML models
- Data engineering or feature engineering
- Hyperparameter tuning details
- Building ML pipelines
- Mathematical/statistical analysis
- Security protocol implementation
- Governance framework development

### Focus Areas for Each Domain
1. **Domain 1 (20%)**: Basic concepts, ML lifecycle, AWS services
2. **Domain 2 (24%)**: GenAI concepts, capabilities, limitations
3. **Domain 3 (28%)**: Foundation models, prompt engineering, RAG
4. **Domain 4 (14%)**: Responsible AI, bias, transparency
5. **Domain 5 (14%)**: Security, compliance, governance

### Key Reminders
- Compensatory scoring (don't need to pass each section)
- No penalty for guessing
- Focus on practical applications, not theory
- Understand when to use AI vs when not to
- Know the business value and trade-offs
- Remember responsible AI principles