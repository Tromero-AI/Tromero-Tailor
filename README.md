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

```python
from tromero_tailor import TailorAI
```

### Initializing the Client

Initialize the TailorAI client using your API keys, which should be stored securely and preferably as environment variables:

```python
client = TailorAI(api_key="your-openai-key", tromero_key="your-tromero-key")
```

### Usage

This class is a drop-in replacement for openai, you should be able to use it as you did before. E.g:

```python
response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": prompt},
    ],
    )
```
And for your trained model

```python
response = client.chat.completions.create(
    model="your-model-name",
    messages=[
        {"role": "user", "content": prompt},
    ],
    )
```
#### Json formatting
Tromero Tailor supports JSON response formatting, allowing you to specify the expected structure of the response using a JSON schema. Formatting works for models you have trained on tromero.

To utilize JSON formatting, you need to define a schema that describes the structure of the JSON object. The schema should conform to the JSON Schema standard. Here is an example schema:
```python
schema = {
    'title': 'Person',
    'type': 'object',
    'properties': {
        'name': {'title': 'Name', 'type': 'string', 'maxLength': 10},
        'age': {'title': 'Age', 'type': 'integer'}
    },
    'required': ['name', 'age']
}
```
##### Specifying the Response Format in API Calls

When making API calls where you expect the response to adhere to a specific format, you can specify the JSON schema using the response_format parameter. Here’s how you can pass this parameter in your API calls:
```python
response_format = {"type": "json_object", "schema": schema}

response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Please provide your name and age."},
    ],
    response_format=response_format
)
```

#### Streaming
Tromero Tailor AI supports streaming responses, which allows you to receive and process data incrementally as it's generated.

##### Enabling Streaming
To enable streaming in your API calls, simply pass the parameter stream=True in your request. This tells the API to return the response incrementally, rather than waiting for the complete response to be ready.

Here's an example of how to initiate a streaming request:
```python
response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Please describe the streaming process."},
    ],
    stream=True
)
```

Once you have initiated a streaming request, you can process the response as it arrives. Each chunk of the response will contain part of the data, and you can handle it within a loop. This is similar to how streaming is handled in the OpenAI API, making it familiar for users transitioning or using both services.

Here’s how you can access and process each chunk of the streamed response:
```python
for chunk in response:
    chunk_message = chunk.choices[0].delta.content
```