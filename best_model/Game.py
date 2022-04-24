from abc import abstractmethod

class Game:
    
    @abstractmethod
    def print_game_state(self, state: str) -> str:
        raise NotImplementedError("No implementation for printing the game state!")

    @abstractmethod
    def vectorize_game_state(self, state: str) -> list:
        raise NotImplementedError("No implementation for vectorizing the game state!")
    
    @abstractmethod
    def get_legal_actions(self, state: str) -> list:
        raise NotImplementedError("No implementation for retreiving the legal actions!")
    
    @abstractmethod
    def get_random_action(self, state: str) -> int:
        raise NotImplementedError("No implementation for finding a random action!")
    
    @abstractmethod
    def get_next_state(self, state: str, action: int) -> str:
        raise NotImplementedError("No implementation for returning the next state!")
    
    @abstractmethod
    def get_reward(self, state: str) -> int:
        raise NotImplementedError("No implementation for returning the reward of the outcome of the game!")
    
    @abstractmethod
    def has_game_ended(self, state: str) -> bool:
        raise NotImplementedError("No implementation for checking the ending conditions of the game!")
