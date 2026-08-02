# Multimodal Forecasting - DSL

This dataset contains **historical stock price data paired with financial news information**.  
Participants are expected to train models using the training dataset and generate predictions for the evaluation period.



## 1. File List

| File                   | Shape           | Size     |
| ---------------------- | --------------- | -------- |
| train.parquet          | (104,400, 38)   | 10.4 MB  |
| test.parquet           | (128,400, 38)   | 12.8 MB  |
| bert_textemb.parquet   | (857,892, 385)  | 1.56 GB  |
| gemini_textemb.parquet | (857,892, 3073) | 12.46 GB |
| lgai_textemb.parquet   | (857,892, 4097) | 16.62 GB |
| linq_textemb.parquet   | (857,892, 4097) | 16.62 GB |
| nvda_textemb.parquet   | (857,892, 4097) | 16.62 GB |
| qwen_textemb.parquet   | (857,892, 4097) | 16.62 GB |





## 2. File Description

### 2-1. train.parquet
This file contains stock price data and paired textual signals for each stock from January 1, 2019 to December 31, 2022.  
Participants should use this dataset to **train their forecasting models**. Each row corresponds to a specific stock and trading date, including price information and references to paired news text IDs.



### 2-2. test.parquet

This file contains stock price data and paired textual signals from January 1, 2019 to December 1, 2023. Participants must use this dataset as input to predict weekday closing prices for the period January 1, 2023 – December 31, 2023. 
*Note: The test input dataset is provided up to December 1, 2023, since December 30–31, 2023 fall on a weekend.*

> ⚠️ **Important constraint**
>
> When predicting a specific date, only information available. **up to 4 weeks. earlier** may be used.
>
> **Example**
>
> - Prediction target: December 29, 2023
> - Maximum usable information: December 1, 2023
>
> Using any information after that point will be considered look-ahead bias.



### 2-3. \*_textemb.parquet

The paired textual signals in `train.parquet` and `test.parquet` do **not contain raw news text**.  
Instead, they contain **text_id references**. The actual embeddings corresponding to each `text_id` can be retrieved from the embedding files. Each embedding file corresponds to a specific text embedding model. Available embedding datasets:
- **bert_textemb.parquet** → [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **lgai_textemb.parquet** → [annamodels/LGAI-Embedding-Preview](https://huggingface.co/annamodels/LGAI-Embedding-Preview)
- **linq_textemb.parquet** → [Linq-AI-Research/Linq-Embed-Mistral](https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral)
- **nvda_textemb.parquet** → [nvidia/llama-embed-nemotron-8b](https://huggingface.co/nvidia/llama-embed-nemotron-8b)
- **qwen_textemb.parquet** → [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
- **gemini_textemb.parquet** → [gemini-embedding-001](https://ai.google.dev/gemini-api/docs/embeddings)





## 4. Handling of Trading Dates

The dataset includes **weekday dates (Monday–Friday)**, and weekend dates are excluded. If a weekday is a market holiday, it is still treated as a trading date in the dataset. In such cases, we make datasets by forward-filling from the previous available trading day.

During prediction, participants must generate predictions for all weekdays, including holidays.  
In other words, submissions must contain predictions for **all 260 weekday dates between January 1, 2023 and December 31, 2023**.



## 5. Column Description

### 5-1. train.parquet / test.parquet

| Column                    | Type     | Description                                            |
| ------------------------- | -------- | ------------------------------------------------------ |
| date                      | datetime | Trading date                                           |
| ticker                    | string   | Stock ticker                                           |
| open                      | float32  | Opening price                                          |
| high                      | float32  | Highest price                                          |
| low                       | float32  | Lowest price                                           |
| close                     | float32  | Closing price                                          |
| volume                    | float32  | Trading volume                                         |
| macro_category\*          | string   | text_id of macroeconomic news                          |
| sector_category\*         | string   | text_id of sector-related news                         |
| relatedCompany_category\* | string   | text_id of news about related companies                |
| targetCompany_category\*  | string   | text_id of news directly related to the target company |
| filing_\*                 | string   | text_id extracted from SEC filing information          |
| lseg_news\*               | string   | text_id of LSEG MRN news tagged for the company        |

*Note: For more details on the pairing methodology for macro_category\* through filing_\*, please refer to [FinTexTS](https://arxiv.org/abs/2603.02702) [1].*



### 5-2. \*_textemb.parquet

| Column  | Type    | Description                                |
| ------- | ------- | ------------------------------------------ |
| text_id | string  | Unique identifier of text                  |
| emb_*   | float32 | Embedding vector corresponding to the text |
