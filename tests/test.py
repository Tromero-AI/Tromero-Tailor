import unittest
from unittest.mock import patch, MagicMock
import datetime
from tromero_tailor import TailorAI
from tromero_tailor.wrapper import MockCompletions
from unittest.mock import patch, MagicMock, mock_open

from pydantic import BaseModel
from typing import List, Literal, Optional

class Message(BaseModel):
    content: str
    role: Literal["assistant", "user", "system"]

class Choice(BaseModel):
    finish_reason: str
    index: int
    logprobs: Optional[dict] = None
    message: Message

class CompletionUsage(BaseModel):
    """Placeholder for usage stats, define as needed"""
    tokens: int

class ChatCompletion(BaseModel):
    id: str
    choices: List[Choice]
    created: int
    model: str
    object: Literal["chat.completion"]
    system_fingerprint: Optional[str] = None
    usage: Optional[CompletionUsage] = None

class TestModel:
    def __init__(self, id):
        self.id = id
        
import unittest
from unittest.mock import patch
from tromero_tailor import TailorAI

# Assuming the classes are defined as you provided above

def create_chat_completion():
    return ChatCompletion(
        id="1234abcd",
        choices=[
            Choice(
                finish_reason='complete',
                index=0,
                message=Message(content="Test response", role="assistant"),
                logprobs=None
            )
        ],
        created=1701070800,
        model="gpt-4",
        object="chat.completion",
        system_fingerprint="example_fingerprint",
        usage=CompletionUsage(tokens=100)
    )

class TestCreateChatCompletion(unittest.TestCase):
    @patch('openai.resources.chat.completions.Completions.create', autospec=True) 
    @patch('openai.resources.models.Models.list', autospec=True)
    @patch('tromero_tailor.wrapper.post_data', autospec=True)
    def test_create_with_openai_model(self, mock_post, mock_list_models, mock_create):
        # Setup the mocked return value for the create method
        mock_create.return_value = create_chat_completion()
        mock_list_models.return_value = [TestModel("valid_model")]
        mock_post.return_value = None

        client = TailorAI(api_key="fake_api_key", tromero_key="fake_tromero_key")
        result = client.chat.completions.create(model="valid_model", messages=["Hello, world!"])
        
        # Assertions to verify behavior
        self.assertIsNotNone(result.choices)
        self.assertEqual(result.choices[0].message.content, "Test response")
        # check post_data is called
        mock_post.assert_called()

    @patch('openai.resources.chat.completions.Completions.create', autospec=True)
    @patch('openai.resources.models.Models.list', autospec=True)
    @patch('tromero_tailor.wrapper.post_data', autospec=True)
    def test_create_with_no_choices_returned(self, mock_post, mock_list_models, mock_create):
        mock_create.return_value = ChatCompletion(
            id="5678efgh",
            choices=[],
            created=1701070800,
            model="gpt-4",
            object="chat.completion",
            system_fingerprint="example_fingerprint",
            usage=CompletionUsage(tokens=50)
        )
        mock_list_models.return_value = [TestModel("valid_model")]
        mock_post.return_value = None

        client = TailorAI(api_key="fake_api_key", tromero_key="fake_tromero_key")
        result = client.chat.completions.create(model="valid_model", messages=["Hello, world!"])
        
        self.assertIsInstance(result, ChatCompletion)
        self.assertEqual(len(result.choices), 0)
        mock_post.assert_not_called()

# Run the test
if __name__ == '__main__':
    unittest.main()



