import functools
import time
from pydantic import BaseModel, Field, computed_field
from enum import Enum
from datetime import datetime, timezone
from nanoid import generate
from typing import List, Dict, Any, Union


class Choice(BaseModel):
    """
    Represents a single choice item with a label for display and an actual value.

    Attributes:
        label (str): Display label for the choice.
        value (str): Actual value of the choice.
    """

    label: str = Field("", description="Display label for the choice")
    value: str = Field("", description="Actual value of the choice")


class BotButtonMessage(BaseModel):
    """
    Represents a message that contains buttons for the user to choose from.

    Attributes:
        text (str): Text to display above the buttons.
        choices (List[Choice]): List of choices for agent reply.
    """

    text: str = Field("", description="Text to show to the user above the buttons")
    choices: List[Choice] = Field(..., description="List of choices for agent reply")
    active: bool = Field(
        True, description="Whether or not the button should be clickable"
    )


class BotDropdownMessage(BaseModel):
    """
    Represents a message that contains a dropdown menu for the user.

    Attributes:
        text (str): Text to display above the dropdown.
        choices (List[Choice]): List of choices for agent reply.
    """

    text: str = Field("", description="Text to show to the user above the dropdown")
    choices: List[Choice] = Field([], description="List of choices for agent reply")
    active: bool = Field(
        True, description="Whether or not the button should be clickable"
    )


class BotHTMLMessage(BaseModel):
    """
    Represents a message that contains HTML content.

    Attributes:
        html (str): HTML string to be rendered for the user.
    """

    html: str = Field("", description="A string of HTML to be rendered for the user")


class BotImageMessage(BaseModel):
    """
    Represents a message that contains an image.

    Attributes:
        url (str): The hosted URL of the image.
    """

    url: str = Field("", description="The image's hosted URL")


class BotTextMessage(BaseModel):
    """
    Represents a basic text message.

    Attributes:
        text (str): Text message content.
        useMarkdown (bool): Flag to determine if text should be rendered using markdown.
    """

    text: str = Field(
        "Hello World", description="A basic text message sent to the user"
    )
    useMarkdown: bool = Field(
        True, description="Whether or not to render text using markdown"
    )


class BotMessageTypes(Enum):
    """
    Enum defining the various types of bot messages.
    """

    text = "text"
    image = "image"
    html = "html"
    button = "button"
    dropdown = "dropdown"


BotPayload = Union[
    BotTextMessage,
    BotImageMessage,
    BotHTMLMessage,
    BotButtonMessage,
    BotDropdownMessage,
]


class BotMessage(BaseModel):
    """
    Represents the main structure of a bot message.

    Attributes:
        type (BotMessageTypes): Type of the message being sent.
        payload (BotPayload): Actual message content/data.
    """

    type: BotMessageTypes = Field(
        "text", description="What type of message is being sent"
    )
    payload: BotPayload = Field("Hello World", description="The message being send")

    @computed_field
    @property
    def messageSize(self) -> int:
        """Compute message size (only works for text messages for now)

        Returns:
            int: message size in characters
        """        
        return 0 if self.type != BotMessageTypes.text else len(self.payload.text)


class Directions(Enum):
    """
    Enum defining the direction of an event.
    """

    incoming = "incoming"
    outgoing = "outgoing"


class Event(BaseModel):
    """
    The base object that gets passed around all the services, capturing event details and context.

    Attributes:
        userId (str): A unique identifier for the user.
        conversationId (str): A unique identifier for the conversation.
        id (str): A unique identifier for this event.
        sentOn (datetime): Timestamp when the event was sent.
        direction (Directions): Direction of the event (either incoming or outgoing).
        payload (Dict[str, Any]): Payload containing the text to be processed.
        botReply (List[BotMessage]): Agent's replies to the user.
    """

    @staticmethod
    def gen_userId():
        """Generate a unique user ID."""
        return generate(size=12)

    @staticmethod
    def gen_convId():
        """Generate a unique conversation ID."""
        return generate(size=14)

    @staticmethod
    def gen_eventId():
        """Generate a unique event ID."""
        return generate(size=18)


    userId: str = Field(
        default_factory=gen_userId, description="A unique identifier for the user"
    )
    conversationId: str = Field(
        default_factory=gen_convId,
        description="A unique identifier for the conversation",
    )
    id: str = Field(
        default_factory=gen_eventId, description="A unique identifier for this event"
    )
    sentOn: datetime = Field(
        default_factory=datetime.utcnow, description="Datetime when the event was sent"
    )
    direction: Directions = Field(
        "incoming", description="Direction of the event - incoming or outgoing"
    )
    payload: Dict[str, Any] = Field(
        {}, description="Payload containing the text to be processed"
    )
    botReply: List[BotMessage] = Field([], description="Agent's replies to the user")

    class Config:
        """Configuration for JSON serialization."""

        json_encoders = {
            datetime: lambda dt: dt.astimezone(timezone.utc).isoformat(),
        }

class BotMetrics(BaseModel):
    """
    Metrics object that captures bot event and inference time info

    Attributes:
        event (Event): Target bot event object.
        startTime (float): Event start time (Unix time).
        endTime (float): Event start time (Unix time).
        success (bool): Indicates whether bot request was successful.
    """    
    event: Event = Field(
        None, description="Target bot event object"
    )
    startTime: float = Field(
        None, description="Event start time (Unix time)"
    )
    endTime: float = Field(
        None, description="Event end time (Unix time)"
    )
    success: bool = Field(
        True, description="Indicates whether bot request was successful"
    )

    @computed_field
    @property
    def responseTime(self) -> float:
        """Get bot response time

        Returns:
            float: bot response time in seconds; 0 if either start or end time is missing
        """        
        if self.startTime is not None and self.endTime is not None:
            return self.endTime - self.startTime
        return 0

    @computed_field
    @property
    def userInputSize(self) -> int:
        """Get user input size

        Returns:
            int: user input size in characters
        """        
        return len(self.event.payload["text"])
    
    @computed_field
    @property
    def botOutputSize(self) -> int:
        """Get bot output message size

        Returns:
            int: bot output message size in characters
        """        
        return functools.reduce(lambda s, r: s + r.messageSize, self.event.botReply, 0)
    
    @computed_field
    @property
    def userId(self) -> str:
        """Get user ID associated with the Event

        Returns:
            str: user ID
        """        
        return self.event.userId

    @computed_field
    @property
    def conversationId(self) -> str:
        """Get conversation ID associated with the Event

        Returns:
            str: conversation ID
        """        
        return self.event.conversationId

    @computed_field
    @property
    def inputMessageType(self) -> str:
        """Get message type associated with the Event

        Returns:
            str: event message type
        """        
        return self.event.payload["type"]

    def trackStart(self):
        """Capture event start
        """        
        self.startTime = time.time()

    def trackEnd(self):
        """Capture event end
        """        
        self.endTime = time.time()

    def captureEvent(self, event: Event):
        """Associate bot event object with this metric instance

        Args:
            event (Event): bot Event object
        """        
        self.event = event