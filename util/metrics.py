from abc import ABC
from prettytable import PrettyTable
from .pydantic_classes import BotMetrics

class PersistMetrics(ABC):
    """Base class for BotMetrics persistence
    """    
    def persist(self, metrics: BotMetrics):
        """Persist BotMetrics object

        Args:
            metrics (BotMetrics): BotMetrics object to persist
        """        
        pass

class ConsoleMetrics(PersistMetrics):
    """Console BotMetrics persistence implementation
    """    
    def persist(self, metrics: BotMetrics):
        """Dumps BotMetrics object into console

        Args:
            metrics (BotMetrics): BotMetrics object to persist
        """        
        t = PrettyTable([
            'When',
            'Success',
            'Conversation ID',
            'User ID',
            'Message type',
            "Input size",
            "Output size",
            "Inference Time"
        ])
        t.add_row([
            round(metrics.startTime, 2),
            "✅" if metrics.success else "🔴",
            metrics.conversationId,
            metrics.userId,
            metrics.inputMessageType,
            metrics.userInputSize,
            metrics.botOutputSize,
            f"⚡ {round(metrics.responseTime, 2)} seconds" 
        ])
        print(t)

class TrackChatbotResponseMetrics(object):
    """BotMetrics context manager
    """    
    def __init__(self, persist: PersistMetrics = ConsoleMetrics()):
        self.persist = persist
    def __enter__(self):
        self.chatbotResponsMetrics = BotMetrics()
        return self.chatbotResponsMetrics
    def __exit__(self, type, value, traceback):
        if type:
            # something went wrong
            self.chatbotResponsMetrics.success = False
        
        self.persist.persist(self.chatbotResponsMetrics)
        return True
