# Tromero Tailor AI

## Installation

To install Tromero Tailor AI, you can use pip.

```
pip install tromero_tailor
```

## Getting Started

Ensure you have set up both your OpenAi key and your Tromero key. You can follow the instructions on our site to create a Tromero key

### Importing the Package

First, import the TailorAI class from the AITailor package:

```
from tromero_tailor import TailorAI
```

### Initializing the Client

Initialize the TailorAI client using your API keys, which should be stored securely and preferably as environment variables:

```
client = TailorAI(api_key="your-openai-key", tromero_key="your-tromero-key")
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
