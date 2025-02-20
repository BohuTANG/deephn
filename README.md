
# Hacker News Podcast Generator

This project automatically fetches top stories from Hacker News, summarizes them using AI, and converts the summaries into podcast-style audio content.

Example: 

https://github.com/user-attachments/assets/6ccc78fc-643f-49ea-a139-6e668706cc1a


## Features

- Fetches latest stories from Hacker News homepage
- Extracts article content and comments using Jina API
- Generates AI summaries using OpenAI/DeepSeek
- Creates bilingual summaries in English and Others
- Converts text summaries to speech using Azure Speech Services
- Saves stories and metadata to JSON files
- Saves audio output as WAV files

## Prerequisites

Service:
1. Jina: https://r.jina.ai
2. DeepSeek v3/OpenAI API
3. Azure Speech Services: https://azure.microsoft.com/en-us/products/ai-services/ai-speech

The following environment variables need to be set:
```
export JINA_KEY=<your-jina-key>
export OPENAI_BASE=<your-openai-base>
export OPENAI_MODEL=<your-openai-model>
export OPENAI_API_KEY=<your-openai-api-key>
export AZURE_SPEECH_KEY=<your-azure-speech-key>
export AZURE_SPEECH_REGION=<your-region>
```

Install:
```
pip install requirments.txt
```

Run:
```
pyhton main.py
```

## Acknowledgements

This project builds upon and is inspired by various open source projects and services. Special thanks to:

- Hacker News API and community
- Jina AI for content extraction
- OpenAI/DeepSeek for AI summarization
- Azure Speech Services for text-to-speech
- All the contributors and maintainers of the libraries used

Your contributions help make projects like this possible.


