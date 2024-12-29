# Real-Time Stock Forecasting with Hybrid Time Series and Machine Learning

## Project Overview
This project integrates time series analysis with machine learning to develop a robust real-time stock price forecasting model. By analyzing historical stock data and using hybrid modeling techniques, the project aims to provide accurate forecasts and actionable insights for financial decision-making.

### Key Highlights
- **Focus Area**: Time Series Forecasting in Financial Applications
- **Primary Models**:
  - **ARIMA** (AutoRegressive Integrated Moving Average)
  - **PROPHET** (Facebook's forecasting model)
  - **XGBR** (Extreme Gradient Boosting Regressor)
  - **SVR** (Support Vector Regressor)
- **Broad Academic Domain**: Financial forecasting, investment strategy optimization, and decision-making support.

---

## Objectives
1. **Accurate Stock Price Prediction**: Enhance forecasting models to achieve better prediction accuracy.
2. **Comparative Model Analysis**: Evaluate ARIMA, PROPHET, XGBR, and SVR models for their strengths and weaknesses.
3. **Real-Time Insights**: Provide traders, businesses, and analysts with actionable trends.
4. **Hybrid Approach**: Combine statistical precision with machine learning adaptability.

---

## Methodology
### 1. Data Collection and Preprocessing
- **Source**: Historical stock market data (past decade).
- **Techniques**:
  - Handle missing values and outliers.
  - Select key features such as closing prices, volume, and trends.

### 2. Model Development
- **ARIMA**: Handles stationary data and linear trends.
- **PROPHET**: Manages seasonality, irregular patterns, and missing values.
- **Hybridization**: Combine ARIMA and PROPHET for enhanced accuracy and flexibility.
- **Support Vector Regressor (SVR)**: For modeling stock trends using support vector machines.
- **XGBR**: Extreme Gradient Boosting for predictive accuracy.

### 3. Evaluation Metrics
- **Root Mean Square Error (RMSE)**
- **Mean Absolute Error (MAE)**

### 4. Insights Generation
- Short-term and long-term market trends for investment strategies.
- Mitigation of limitations in traditional models, such as sudden market shocks, with LSTM networks.

---

## Technical Architecture and Tools

### 1. **MongoDB Integration**
MongoDB is used to store and retrieve stock market data, including historical price data, predictions, and model-related metadata. This NoSQL database is ideal for handling large volumes of data, which is essential for this project due to the high frequency and large size of stock data.

- **MongoDB URI**: Stored in the environment variables to ensure secure connection (`MONGO_URI`).
- **Collection**: `StockData` collection stores information related to stock prices and prediction results.
- **Operations**:
  - **Data Insertion**: Historical stock data is inserted into MongoDB from external sources.
  - **Data Retrieval**: The model fetches relevant historical stock data to make predictions.
  - **Model Data Storage**: Model predictions and performance metrics are saved in MongoDB for future analysis.

### 2. **Kafka for Real-Time Prediction Messaging**
Kafka is utilized for real-time data processing and message queuing. Once the stock price predictions are made, the results are sent to a Kafka topic, enabling downstream consumers (such as trading systems, analytics tools, or alerting systems) to consume the predictions in real time.

- **Kafka Producer**: The Flask API serves as a Kafka producer, sending prediction results to the Kafka topic.
- **Kafka Topics**: Predictions are sent to specific topics based on company ID or prediction type.
- **Use Case**: For example, a trading system can subscribe to the Kafka topic and execute trades based on the predicted stock prices.

### 3. **Model Deployment and Real-Time Inference**
The models (LSTM, SVR, XGBR) are loaded and utilized for real-time predictions through the Flask API. When a new set of stock data is received via the `/predictStockData` endpoint, the API triggers model inference using the hybrid models.

- **LSTM Model**: Used for capturing long-term dependencies and trends in stock data.
- **SVR Model**: Support Vector Regressor provides a robust approach for trend prediction.
- **XGBR Model**: Extreme Gradient Boosting Regressor is used for capturing non-linear relationships in the data.
- **Ensemble Model**: Combines the predictions from LSTM, SVR, and XGBR to create a final prediction.

### 4. **Scalable Model and Prediction System**
- **Microservices Architecture**: Each model (LSTM, SVR, XGBR) is encapsulated in separate modules that interact with the Flask API. This approach makes the system scalable and easy to maintain.
- **Real-Time Prediction**: The Flask API ensures real-time predictions for users, with the ability to make multiple predictions for different companies concurrently.

### 5. **Model Versioning and Updates**
- **Model Storage**: Trained models are stored in a directory structure (`models/{model_type}/{company_id}.pkl` or `.h5`), and can be versioned by renaming files or storing in cloud storage like S3.
- **Dynamic Model Loading**: The `load_models` function dynamically loads the appropriate models (LSTM, SVR, XGBR, ensemble) based on the company ID, ensuring that predictions are accurate and up-to-date.

---

## Planned Workflow

| Phase                        | Key Activities                                       | Timeline               |
|------------------------------|-----------------------------------------------------|------------------------|
| Define Abstract              | Research goals and scope definition                 | 25th Nov - 8th Dec      |
| Data Collection              | Historical data collection and trend analysis       | 9th Dec - 22nd Dec      |
| Design Workflow              | Preprocessing pipeline and model selection          | 23rd Dec - 6th Jan      |
| Initial Development          | Data preprocessing and basic model testing          | 7th Jan - 19th Jan      |
| Mid-Semester Review          | Report preparation and feedback integration         | 15th Jan - 10th Feb     |
| Model Refinement             | Feature expansion and validation                    | 11th Feb - 24th Feb     |
| Documentation                | Final project report and methodology details        | 26th Feb - 2nd Mar      |
| Final Submission & Evaluation| Presentation and submission                         | Post 8th Mar            |

---

## Coding Language and Tools Plan

### Programming Languages
- **Python**: Primary language for data preprocessing, model development, and evaluation.
- **R**: For statistical analysis and time series visualization (if required).

### Libraries and Frameworks
- **Data Handling**:
  - `pandas`: For data manipulation and analysis.
  - `numpy`: For numerical computations.
- **Visualization**:
  - `matplotlib` and `seaborn`: For plotting data and results.
- **Time Series Modeling**:
  - `statsmodels`: For ARIMA implementation.
  - `fbprophet`: For PROPHET modeling.
- **Machine Learning**:
  - `scikit-learn`: For model evaluation and additional ML tasks.
  - `xgboost`: For implementing the XGBR model.
- **Deep Learning**:
  - `tensorflow` or `keras`: For potential LSTM development.
- **Model Validation**:
  - `sklearn.metrics`: For computing RMSE and MAE.

### Tools and Platforms
- **Jupyter Notebook**: For iterative development and experimentation.
- **Git/GitHub**: For version control and collaboration.
- **Cloud Platforms**:
  - AWS S3: For storing datasets.
  - Google Colab or AWS EC2: For running experiments with computational resources.

---

## Benefits and Applications
- **For Investors**: Enable informed decision-making with accurate predictions.
- **For Businesses**: Optimize resource allocation based on market trends.
- **For Researchers**: Provide a foundation for hybrid forecasting methods in financial applications.

---

## Keywords
- Stock Price Prediction
- Time Series Analysis
- ARIMA
- PROPHET
- Financial Forecasting
- Market Trends

---

## References
1. Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice* (2nd ed.).
2. Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.).
3. Zhang, G. P. (2003). "Time series forecasting using a hybrid ARIMA and neural network model." *Neurocomputing*, 50, 159-175.
4. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). "Statistical and Machine Learning forecasting methods: Concerns and ways forward." *PLOS ONE*, 13(3), e0194889.
5. Chakraborty, T., & Bhattacharyya, S. (2020). "Deep learning in stock market prediction: A survey." *Information Systems and Operational Research*.

---
