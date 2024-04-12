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