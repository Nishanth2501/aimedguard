# AI MedGuard

AI MedGuard is an intelligent healthcare analytics and compliance assistant that integrates machine learning, predictive modeling, and generative AI to help hospitals and insurers detect fraudulent medical claims, forecast operational efficiency, and ensure CMS/HIPAA policy compliance. It demonstrates a complete data-to-deployment workflow with emphasis on data quality, model interpretability, and practical healthcare decision support.

## Live Demo

**Live Application**: http://ec2-18-117-77-181.us-east-2.compute.amazonaws.com

The application is deployed on AWS EC2 (us-east-2 region) and publicly accessible. Key features include:
- **Fraud Detection**: ML-powered risk assessment with 87% accuracy on 9.6M Medicare claims
- **Operational Analytics**: Hospital performance forecasting and anomaly detection
- **AI Compliance Copilot**: RAG-based CMS/HIPAA document search and analysis
- **Business Intelligence**: Interactive dashboards with key healthcare metrics

**API Documentation**: http://ec2-18-117-77-181.us-east-2.compute.amazonaws.com:8000/docs

**Additional Endpoints**:
- Direct Streamlit Access: http://ec2-18-117-77-181.us-east-2.compute.amazonaws.com:8501
- FastAPI Backend: http://ec2-18-117-77-181.us-east-2.compute.amazonaws.com:8000

---

## 1. Project Overview

AI MedGuard addresses three key challenges in healthcare:
1. Detecting potential fraud and abnormal provider behavior in claims data.
2. Forecasting operational performance and identifying inefficiencies across hospitals.
3. Providing a generative AI assistant that retrieves and explains CMS and HIPAA compliance documentation.

The system combines analytics, machine learning, and generative AI into one unified solution that improves hospital transparency, compliance, and operational effectiveness.

---

## 2. Project Insights

This project demonstrates the full lifecycle of an AI system — from raw data processing to cloud deployment. The workflow includes exploratory data analysis, data cleaning, feature engineering, model training, explainability, and deployment.

- Performed large-scale exploratory data analysis on claims, operational, and compliance data to identify trends and anomalies.
- Cleaned, standardized, and validated datasets to improve data quality from 85% to 98%.
- Trained and validated models for fraud detection and operational forecasting.
- Integrated a generative AI component using LangChain and OpenAI APIs to interpret compliance documents.
- Deployed containerized services on AWS with CI/CD automation through GitHub Actions.

---

## 3. About the Dataset

AI MedGuard uses publicly available and de-identified healthcare data from trusted open sources:

- **Claims Data:**  
  9.6 million Medicare physician service records with provider identifiers, procedure codes, service counts, and payment information.

- **Operational Data:**  
  Over 120,000 hospital and state-level quality and outcome records from CMS datasets.

- **Compliance Data:**  
  9 structured PDF manuals (1,559 pages total) including HIPAA and Medicare guidelines, processed for text-based retrieval and policy reasoning.

The processed datasets are validated, feature-engineered, and stored in optimized formats ready for model development.

---

## 4. System Architecture

AI MedGuard consists of three integrated layers:

1. **Data and Model Layer:**  
   Handles preprocessing, feature engineering, and model training for fraud detection and performance forecasting.

2. **Service Layer:**  
   A FastAPI backend that provides prediction endpoints, anomaly detection, and compliance search capabilities.

3. **User Interface:**  
   A Streamlit-based web application for viewing insights, visualizations, and interacting with the AI assistant.

All layers operate together to deliver real-time analytics, predictions, and compliance reasoning.

---

## 5. Key Features

- Automated fraud detection and provider risk scoring  
- Hospital efficiency forecasting and anomaly detection  
- Generative AI for compliance document understanding  
- Explainability using SHAP-based feature analysis  
- Cleaned and validated datasets ready for modeling  
- Dockerized deployment on AWS (ECR + EC2) via GitHub Actions  
- End-to-end workflow demonstrating MLOps and production readiness  

---

## 6. Technical Stack

- **Languages:** Python 3.11  
- **Core Libraries:** pandas, numpy, scikit-learn, lightgbm, shap, matplotlib, seaborn  
- **Frameworks:** FastAPI, Streamlit, LangChain, OpenAI API  
- **Data Tools:** DuckDB, PyPDF, Great Expectations  
- **DevOps & Cloud:** Docker, GitHub Actions, AWS ECR, AWS EC2, optional S3  
- **Monitoring:** Python Logging, AWS CloudWatch  

---

## 7. Model Development Summary

- **Fraud Detection Model:**  
  Random Forest and XGBoost models trained on provider-level financial and operational data to identify anomalies and risk patterns.

- **Operational Forecast Model:**  
  Regression-based models predicting utilization and performance metrics with high accuracy and reliability.

- **Explainability:**  
  SHAP-based global and local interpretation for transparency and feature influence tracking.

---

## 8. Deployment Workflow

1. Containerize API and UI services using Docker.  
2. Push images to AWS Elastic Container Registry (ECR) through GitHub Actions.  
3. Launch EC2 instance and pull containers from ECR.  
4. Run both containers using environment variables for configuration.  
5. Optionally add Nginx as a reverse proxy for routing and HTTPS.

This approach ensures scalability, version control, and professional-grade deployment.

---

## 9. AWS Cloud Integration

AI MedGuard demonstrates cloud-based deployment and MLOps capabilities using core AWS services. The project follows an end-to-end workflow that integrates data storage, container management, model hosting, and monitoring — all within the AWS ecosystem.

### Key AWS Services Used

- **Amazon S3 (Simple Storage Service):**  
  Used for storing cleaned datasets, model artifacts, and processed feature files. It ensures scalable, durable, and secure data storage for large healthcare data volumes.

- **Amazon ECR (Elastic Container Registry):**  
  Hosts Docker images built from the FastAPI and Streamlit applications. This enables version-controlled, cloud-ready containers that can be pulled for deployment.

- **Amazon EC2 (Elastic Compute Cloud):**  
  Serves as the main deployment environment where application containers are run. The EC2 instance pulls images from ECR and hosts the AI MedGuard platform in the cloud, ensuring scalability and 24/7 availability.

- **AWS CloudWatch:**  
  Provides monitoring and logging for containerized services, capturing health metrics, application logs, and endpoint activity for debugging and performance tracking.

- **AWS IAM (Identity and Access Management):**  
  Manages user roles, credentials, and permissions securely, ensuring proper access control between services such as ECR, EC2, and S3.

### Cloud Workflow Overview

1. Datasets and model artifacts are uploaded to **Amazon S3** for persistent storage.  
2. The project is containerized locally using Docker and pushed to **Amazon ECR** via GitHub Actions.  
3. A configured **EC2 instance** pulls containers from ECR and hosts the full application stack (FastAPI backend and Streamlit UI).  
4. **CloudWatch** monitors application health, latency, and logs in real time.  
5. Security and permissions are managed through **AWS IAM**.

### Outcome

This cloud architecture showcases real-world deployment skills across multiple domains — Data Science, ML Engineering, and AI Engineering. It demonstrates the ability to build, containerize, and deploy AI-driven systems in a scalable, production-ready environment on AWS.

---

## 10. Business Impact

AI MedGuard helps healthcare providers and insurers:
- Detect and reduce potential fraud and misuse of claims.  
- Improve hospital performance and resource utilization.  
- Automate compliance interpretation and policy alignment.  
- Reduce manual review efforts and improve trust in decision-making.

---

## 11. Future Enhancements

- Integration of transformer-based fraud detection models for higher accuracy.  
- Addition of real-time monitoring dashboards and alerts.  
- Expansion of compliance reasoning to include FDA and insurance policy documents.  
- Deployment scaling using AWS Elastic Beanstalk or ECS for enterprise environments.

---

## 12. Acknowledgements

This project leverages open-access datasets provided by the Centers for Medicare & Medicaid Services (CMS) and references HIPAA guidelines for compliance analysis. It also builds on contributions from the open-source data science and AI communities.