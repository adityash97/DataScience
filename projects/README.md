# 📊 PandasAI Clone using LangGraph

This project replicates the core idea of **PandasAI** using **LangGraph + LLMs**, enabling users to interact with datasets using natural language queries.

---

## 🚀 Overview

The system allows users to:

* Ask questions in **plain English**
* Automatically generate **Pandas code using LLM**
* Execute the generated code on a dataset
* Handle errors intelligently using **retry loops**
* Return a **human-readable summarized response**

---

## 🧠 How It Works

This project uses a **LangGraph-based workflow** to orchestrate multiple steps:

1. **User Query → LLM**

   * Converts natural language into Pandas code

2. **Code Execution**

   * Executes generated code using `exec()`

3. **Error Handling Loop**

   * If execution fails → LLM is re-invoked with error context

4. **Response Summarization**

   * Converts raw output into a clean human-readable answer

---

## 🔄 LangGraph Flow

```
START
  ↓
generate_llm_response
  ↓
execute_query
  ↓
   ├── (Error) → call_llm_again → execute_query
   └── (Success) → summarize_response
  ↓
END
```

---

## 📦 Tech Stack

* **Python**
* **LangGraph**
* **LangChain**
* **OpenAI (GPT-4o-mini)**
* **Pandas**

---

## 📁 Dataset

* CSV file (`sample_100mb.csv`)
* Example columns:

  * `CustId`
  * `Rating`
  * `Date`
  * `MovieId`
  * `ReleaseYear`
  * `MovieTitle`

---

## ⚙️ Installation

```bash
pip install langchain langchain-openai langgraph pandas python-dotenv
```

---

## 🔑 Environment Setup

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## ▶️ Usage

### 1. Load Dataset

```python
movie_data_set = pd.read_csv('./sample_100mb.csv')
movie_data_set.drop(columns=['Unnamed: 0'], inplace=True)
```

---

### 2. Define Initial State

```python
init_state = {
    'user_query': 'Give me 5 rows of the dataset',
    'columns': list(movie_data_set.columns),
    'dataset_name': 'movie_data_set',
    'llm_response': '',
    'user_to_response': [{}],
    'error_message': None,
    'retry_on_error': 2
}
```

---

### 3. Run Graph

```python
graph.invoke(init_state)
```

---

## 🧩 Core Components

### 1. **State Management (`DataState`)**

Tracks:

* User query
* Dataset metadata
* LLM response
* Execution result
* Error handling

---

### 2. **LLM Prompting**

* Generates **pure Python code only**
* No formatting, no explanations
* Uses dataset schema context

---

### 3. **Execution Engine**

```python
exec(f"result = {state['llm_response']}", {}, local_vars)
```

* Runs dynamically generated code
* Captures results as dictionary

---

### 4. **Retry Mechanism**

* If execution fails:

  * Error is passed back to LLM
  * Code is regenerated
* Controlled via `retry_on_error`

---

### 5. **Summarization Layer**

* Converts raw output into user-friendly explanation

---

## ✅ Example Queries

* "Give me 5 rows of the dataset"
* "Show last row"
* "Average rating by movie"
* "Top 10 highest rated movies"

---

## ⚠️ Limitations

* Uses `exec()` → **not safe for production**
* LLM may generate incorrect code
* No sandboxing
* Performance depends on dataset size

---

## 💡 Future Improvements

* Replace `exec()` with safe execution environment
* Add SQL support
* Integrate vector DB for semantic queries
* Add caching layer
* UI interface (Streamlit / React)

---

## 🎯 Use Cases

* Data exploration tools
* AI-powered analytics dashboards
* Internal business intelligence tools
* Natural language querying over datasets

---

## 🧠 Key Insight

> This project demonstrates how LLMs + workflow orchestration (LangGraph) can transform natural language into executable data pipelines.

---

## 👨‍💻 Author

Aditya Anand

* GitHub: https://github.com/adityash97

---

## ⭐ If you found this useful

Give it a ⭐ and consider building on top of it!

---
