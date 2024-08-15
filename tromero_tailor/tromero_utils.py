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

def tags_to_string(tags):
    return ','.join(tags)
        

def validate_file_content(file_path):
    # Check if the file extension is .jsonl
    if not file_path.endswith('.jsonl'):
        print("Error: File is not a .jsonl file.")
        return

    with open('sample_data_1000.jsonl', 'r') as file:
        content = file.read()

    lines = content.split('\n')
    index = 0
    has_error = False
    total_lines = 0

    def validate_line(index, total_lines):
        nonlocal has_error
        while index < len(lines) and not has_error:
            line = lines[index].strip()
            if line:
                try:
                    json_data = json.loads(line)
                    if 'messages' not in json_data or not isinstance(json_data['messages'], list):
                        raise ValueError(f'Invalid format on line {index + 1}: "messages" should be an array.')

                    has_user = False
                    has_assistant = False
                    roles = []
                    for message in json_data['messages']:
                        if 'role' not in message or 'content' not in message:
                            raise ValueError(f'Invalid format on line {index + 1}: Each message should have a "role" and "content".')
                        if message['role'] == 'user':
                            has_user = True
                        if message['role'] == 'assistant':
                            has_assistant = True
                        if message['role'] not in ['system', 'user', 'assistant']:
                            raise ValueError(f'Invalid role on line {index + 1}: Each message role should be either "system", "user", or "assistant".')
                        roles.append(message['role'])

                    if not has_user:
                        raise ValueError(f'Invalid format on line {index + 1}: Missing "user" message.')
                    if not has_assistant:
                        raise ValueError(f'Invalid format on line {index + 1}: Missing "assistant" message.')
                    
                    for i in range(1, len(roles)):
                        if roles[i] == roles[i - 1]:
                            raise ValueError(f'Invalid format on line {index + 1}: Roles should alternate starting with "user".')
                    if roles[0] not in ['user', 'system']:
                        raise ValueError(f'Invalid format on line {index + 1}: The first role should be "user" or "system".')
                    if roles[0] == 'system' and roles[1] != 'user':
                        raise ValueError(f'Invalid format on line {index + 1}: The role following "system" should be "user".')
                except ValueError as e:
                    print(str(e))
                    has_error = True
                    break
                except json.JSONDecodeError:
                    print(f"Error parsing JSON on line {index + 1}")
                    has_error = True
                    break
                total_lines += 1
            index += 1
        return total_lines

    total_lines = validate_line(index, total_lines)
    if not has_error:
        return True
    else:
        print("Validation encountered errors.")
        return False
