# Tromero Closed AI

## Installation

To install Tromero Closed AI, you can use pip.

```
pip install tromero
```

## Getting Started
### Importing the Package

First, import the ClosedAi class from the AITailor package:

```
from tromero import ClosedAI
```

### Initializing the Client

Initialize the ClosedAi client using your API keys, which should be stored securely and preferably as environment variables:

```
client = ClosedAi(api_key="your-openai-key", tromero_key="your-tromero-key")
```

### Usage

This class is a drop-in replacement for openai, you should be able to use it as you did before. E.g:

```
response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": prompt},
    ],
    temperature=0.5,
    )
```