# ChatPDF - AI-Powered Document Chat

A modern web application that enables conversational interactions with document content using Chainlit and AI. This application provides an intuitive chat interface for querying and extracting information from documents.

## Features

- **Document Q&A**: Ask questions about your documents and get AI-powered answers
- **Semantic Search**: Find relevant information across documents using advanced embeddings
- **Source Citations**: All answers include citations to the source documents
- **Chat Memory**: Maintains conversation context for natural follow-up questions
- **Interactive Chat Interface**: User-friendly conversational interface powered by Chainlit
- **Responsive Design**: Works on desktop and mobile devices

## Prerequisites

- Python 3.10 or higher
- Poetry (Python package manager)
- OpenAI API key for AI-powered features
- Pinecone API key for vector database functionality

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd chatpdf
```

2. Install dependencies using Poetry:

```bash
make install
```

3. Create a `.env` file with your API keys and configuration:

```bash
cp .env.example .env
```

Edit the `.env` file to add your API keys:

```
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

## Running the Application

Start the application with:

```bash
make run
```

The application will be available at `http://localhost:7860` in your web browser.

## Usage

1. Open the application in your web browser
2. Type your questions about documents in the chat interface
3. The AI will search through the documents and provide answers with citations
4. Continue the conversation with follow-up questions for further information

## Project Structure

- `app.py`: Main application file with Chainlit interface and AI logic
- `examples_manager.py`: Manages example queries and responses
- `prompt_manager.py`: Handles system prompts for AI components
- `logging_config.py`: Configuration for application logging
- `prompts/`: Contains prompt templates for AI interactions
- `examples/`: Sample queries and responses for AI training
- `public/`: Static assets for the web interface

## Available Commands

- `make install`: Install all required dependencies using Poetry
- `make run`: Start the application on port 7860
- `make clean`: Clean up cache files and other temporary data

## Dependencies

- Chainlit: Framework for building conversational AI applications
- OpenAI: For AI capabilities and embeddings
- Pinecone: Vector database for semantic search
- Pydantic: Data validation
- TikToken: Token counting for LLM input

## Troubleshooting

If you encounter issues:

1. Ensure all dependencies are installed correctly
2. Verify your API keys in the `.env` file are valid
3. Check the logs directory for detailed error information
4. Make sure port 7860 is available on your system

## License

[License information]

## Contributing

[Contribution guidelines] 