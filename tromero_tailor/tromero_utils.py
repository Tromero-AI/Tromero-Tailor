import json
import re
class Message:
    def __init__(self, content, role="assistant"):
        self.content = content
        self.role = role

class Choice:
    def __init__(self, message):
        self.message = Message(message)

class Response:
    def __init__(self, choices):
        self.choices = choices

def mock_openai_format(messages):
    choices = [Choice(messages)]  # Create a list of Choice objects
    response = Response(choices)
    return response

class StreamChoice:
    def __init__(self, message):
        self.delta = Message(message)

def mock_openai_format_stream(messages):
    choices = [StreamChoice(messages)]  # Create a list of Choice objects
    response = Response(choices)
    return response

class StreamResponse:
    def __init__(self, response):
        self.response = response

    def __iter__(self):
        for chunk in self.response.iter_content(chunk_size=1024):
            # chunk_dict = json.loads(chunk)
            chunk = chunk.decode('utf-8')
            json_str = chunk[5:]
            pattern = r'\"token\":({.*?})'
            match = re.search(pattern, json_str)   
            if match:
                json_str = match.group(1)
            else:
                break
            chunk_dict = json.loads(json_str)
            formatted_chunk = mock_openai_format_stream(chunk_dict['text'])
            yield formatted_chunk
