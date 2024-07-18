import json
import re
class Message:
    def __init__(self, content, role="assistant"):
        self.content = content
        self.role = role

class Choice:
    def __init__(self, message):
        self.message = Message(message)

class Usage:
    def __init__(self, usage):
        self.completion_tokens = usage['completion_tokens']

class Response:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage
        

def mock_openai_format(messages, usage):
    choices = [Choice(messages)]  # Create a list of Choice objects
    usage = Usage(usage)
    response = Response(choices, usage)
    return response

class StreamChoice:
    def __init__(self, message):
        self.delta = Message(message)

def mock_openai_format_stream(messages):
    choices = [StreamChoice(messages)]  # Create a list of Choice objects
    response = Response(choices)
    return response    
        
