<br />
<div align="center">
  <h2 align="center">Question-Answering API with Custom Response Handler</h2>
</div>

<br/>

## Overview

This API serves as a question-answering interface for questions on the german law, leveraging a backend that processes queries and returns answers sourced from a database made out of german law books. The API is designed to receive questions through a RESTful endpoint and provide answers in a conversational format, mimicking a dialogue with a virtual character named "Harvey Specter" from the series suits. Each response includes the answer content, the sender's name, and a timestamp, enriching the user experience with a personalized touch.

<br />
<div align="center">
    <img src="Images/RAG.pdf" alt="Retrieval Augmented Generation" width="70%">
</div>
<br/>


## Features

- **Question-Answering Capability:** Processes textual queries and provides accurate answers by consulting an extensive database.
- **Conversational Responses:** Returns answers in a conversational format, attributed to the virtual character "Harvey Specter."
- **Timestamped Replies:** Each response is accompanied by a timestamp, adding context to the dialogue flow.

## Installation

Follow these steps to set up the API server:

1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the cloned directory:
   ```
   cd <repository-directory>
   ```
3. Install the required dependencies
4. Launch the API server:
   ```
   python app.py
   ```

Ensure Python 3.9+ and pip are installed on your system before proceeding.

## Usage

### Making a POST Request to `/query`

To interact with the API, you'll need to send a POST request to the `/query` endpoint with a JSON payload containing your question.

#### Request Format

- **Endpoint:** `/query`
- **Method:** POST
- **Headers:**
  - Content-Type: application/json
- **Body:**
  - `message`: The text of your question.

#### Example Request

Here's how to use `curl` to send a question to the API:

```sh
curl -X POST http://localhost:5000/query \
     -H "Content-Type: application/json" \
     -d '{"message": "What is the theory of relativity?"}'
```

#### Response Format

The API will return a JSON object that includes the answer to your question, the name of the sender (Harvey Specter), and a timestamp marking the response time.

```json
{
  "content": "The theory of relativity, proposed by Albert Einstein, explains the interrelations of time and space...",
  "sender": "Harvey Specter",
  "timestamp": "2024-02-22T12:34:56.789Z"
}
```

## Support

For any questions or issues, please open an issue in the GitHub repository or contact the project maintainers directly.