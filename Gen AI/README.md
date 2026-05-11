# AI Chatbot with Mistral AI 🤖

A simple conversational AI chatbot built using LangChain and Mistral AI with both:

- Terminal-based chat interface
- Streamlit UI chatbot
- Conversation memory/history support
- Structured response formatting experiments

---

## 📂 Project Structure

```bash
Gen AI/
│
├── chatbot/
│   ├── chatbot_terminal.py
│   ├── chatbot_streamlit_ui.py
│
├── chatbot_formatted_output/
│   ├── chatbot_terminal.py
│   ├── chatbot_streamlit_ui.py
│   ├── data_model.py
│
└── mistral_chat_history.json
```

---

# 🚀 Features

- Conversational chatbot using Mistral AI
- Chat memory/history support
- Streamlit UI interface
- Terminal-based chatbot
- Environment variable support using `.env`
- Structured output formatting using LangChain
- Saves conversation history into JSON
- Funny assistant personality with emoji support 😄

---

# 🛠️ Tech Stack

- Python
- LangChain
- Mistral AI
- Streamlit
- dotenv

---

# 📦 Installation

## 1. Clone Repository

```bash
git clone <your_repo_url>
cd <repo_name>
```

---

## 2. Create Virtual Environment

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

OR manually:

```bash
pip install langchain langchain-mistralai python-dotenv streamlit
```

---

# 🔑 Environment Setup

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_api_key_here
```

Get your API key from:

https://console.mistral.ai/

---

# ▶️ Run Terminal Chatbot

```bash
python chatbot_terminal.py
```

Example:

```text
You : Hello
Mistral : Hey there 😄 How's your day going?
```

Type:

```text
exit
```

to stop the chatbot.

---

# 🖥️ Run Streamlit UI

```bash
streamlit run chatbot_streamlit_ui.py
```

Then open:

```text
http://localhost:8501
```

---

# 🧠 How Memory Works

The chatbot stores previous conversations inside:

```python
history = []
```

Each interaction is appended as:

```python
history.append(humanMessage)
history.append(AIMessage(content=response.content))
```

When the user exits, history is saved into:

```text
mistral_chat_history.json
```

---

# ⚙️ Core LangChain Components Used

## Chat Model

```python
ChatMistralAI
```

Used for interacting with Mistral LLM.

---

## Message Types

```python
SystemMessage
HumanMessage
AIMessage
```

These help structure conversation roles.

---

# 🎭 System Prompt

```python
"You are a funny assistant with a good memory of previous conversations."
```

This defines chatbot personality and behavior.

---

# 📘 Learning Concepts Covered

This project helps understand:

- LLM integrations
- Prompt engineering
- Conversation memory
- Chat history management
- LangChain message architecture
- Streamlit UI integration
- Structured outputs from LLMs
- Environment variable management

---

# 🔮 Future Improvements

- Database-based memory
- Vector database integration
- RAG (Retrieval-Augmented Generation)
- Authentication system
- Multi-user chat sessions
- Voice chatbot support
- File upload support
- Agent/tool calling

---

# 📸 Project Modules

## `chatbot/`

Basic chatbot implementation:

- Terminal chatbot
- Streamlit chatbot UI

---

## `chatbot_formatted_output/`

Advanced chatbot experiments:

- Structured responses
- Data models
- Output formatting

---

# 🧪 Sample Workflow

```text
User Input
    ↓
LangChain Message Formatting
    ↓
Mistral AI Model
    ↓
AI Response
    ↓
Conversation History Storage
```

---

# 📄 Example Code Flow

```python
response = llm.invoke(message)
```

LangChain sends all formatted messages to Mistral AI and receives the generated response.

---

# 👨‍💻 Author

Aditya Anand

- GitHub: https://github.com/adityash97

---

# ⭐ Repository Goal

This repository is part of my learning journey in:

- Generative AI
- LangChain
- LLM Applications
- AI-powered chat systems
- Full Stack AI Development
