# Assignment 2

The goal of this assignment is to design and implement an AI system with a conversational interface.

The Assignment-2 is Weather chat-based AI assistant built in Python jupyter notebook using UV Environment setup. It is designed with OpenAI's GPT-4o-mini alongside local vector search embeddings and external WeatherAPI from api.weatherapi.com. The assistant operates under a highly specific, analytical persona and features a dynamic routing system that autonomously decides when to fetch real-world data, when to query its local database, and when to respond conversationally. The city facts have been specifically added under ChromaDB for query embedding to provide fun facts about the city along with the weather.

# Requirements

Assignment 2 followed and met all the following specifications.

## Services

You must include at least **three services** in your system.

### Service 1: API Calls

Service 1: Live API Integration (Weather Data)
The system integrates with WeatherAPI.com to pull real-time meteorological and astronomical data. Instead of presenting raw JSON to the user, the assistant extracts targeted variables (temperature, wind, moon phase, etc.) and uses the LLM to synthesize a natural, conversational summary of the environmental conditions.

### Service 2: Semantic Query

Service 2: Semantic Query & Vector Database (ChromaDB)
The application utilizes a local, persistent ChromaDB instance to handle semantic search. A lightweight dataset consisting of Canadian city facts is embedded using OpenAI's text-embedding-3-small model. This allows users to query the database using conceptual questions (e.g., "Which city is best for biking?"), which the system mathematically matches to the most relevant hardcoded fact.

### Service 3: Function Calling)

Service 3: Intelligent Router (Function Calling)
Serving as the "brain" of the application, Service 3 uses OpenAI's Function Calling capabilities. The LLM is provided with a toolkit defining the Weather API and ChromaDB functions. It analyzes the user's intent and automatically routes the query to the correct external script, processes the returned data, and generates a unified response.

  * [Function Calling](https://platform.openai.com/docs/guides/function-calling) (API calling is acceptable, but not mandatory)


## User Interface

* The system must include a chat-based interface, preferably implemented with Gradio.
* Give the chat client a distinct personality to make the interaction engaging. For example, assign a specific tone, role, or conversational style.
* The chat interface must maintain memory throughout the conversation.

System Features
•	Distinct Persona: The assistant is prompted to act as a results-driven Analytics and Machine Learning Engineer. Its tone is strictly professional, precise, and analytical.
•	Contextual Memory Management: To maintain long-term chat stability and prevent context-window overflow, the Gradio interface utilizes a sliding-window memory system. It actively tracks the conversation but only feeds the most recent four exchanges to the LLM.

## Guardrails and Other Limitations

Strict Security Guardrails: The system prompt enforces strict topic limitations. The assistant will politely but firmly refuse to discuss restricted subjects (cats, dogs, horoscopes, and Taylor Swift). It also includes prompt-injection defense, refusing any user instructions to reveal or alter its core rules.

* The model must not respond to questions on certain restricted topics:

  * Cats or dogs
  * Horoscopes or Zodiac Signs
  * Taylor Swift

## Implementation

+ Implement your code in the folder `./05_src/assignment_chat`.

System Workflow
The typical lifecycle of a user interaction follows a specific, multi-pass workflow:
	1.	Input & Context Assembly: The user submits a prompt via the Gradio web interface. The system bundles this prompt with the core system instructions and the sliding-window chat history.
	2.	First Evaluation Pass: The packaged context is sent to the LLM. The model evaluates the prompt against its available tools to determine if external data is required.
	3.	Tool Execution (If Applicable): * If the user asks about the weather, the LLM requests a weather check. The local Python script intercepts this, runs the requests.get call to WeatherAPI, and returns the data.
•	If the user asks for trivia, the LLM requests a database search. The local Python script queries ChromaDB for a semantic match and retrieves the string.
	4.	Second Synthesis Pass: The newly acquired data is appended to the message history and sent back to the LLM.
	5.	Final Output: The LLM processes the raw tool output, enforces its persona and guardrails, and returns a natural-language response to the Gradio interface for the user to read.
