"""
Main Euchre Game Engine
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from .card import Card, Suit
from .deck import Deck
from .player import Player, PlayerType
from .trick import Trick


class GamePhase(Enum):
    """Phases of a Euchre game"""

    SETUP = "setup"
    DEALING = "dealing"
    TRUMP_SELECTION_ROUND1 = "trump_selection_round1"  # Can pick up card
    TRUMP_SELECTION_ROUND2 = (
        "trump_selection_round2"  # Can call any suit except turned up
    )
    DEALER_DISCARD = "dealer_discard"  # Dealer must discard after picking up
    PLAYING = "playing"
    HAND_COMPLETE = "hand_complete"
    GAME_OVER = "game_over"


class GameState:
    """Represents the current state of a Euchre game"""

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.phase = GamePhase.SETUP
        self.players: List[Player] = []
        self.deck = Deck()
        self.dealer_position = 0
        self.current_player_position = 0

        # Trump selection
        self.turned_up_card: Optional[Card] = None
        self.trump: Optional[Suit] = None
        self.trump_caller_position: Optional[int] = None
        self.going_alone = False
        self.alone_player_position: Optional[int] = None

        # Score tracking
        self.team1_score = 0  # Players 0 & 2
        self.team2_score = 0  # Players 1 & 3
        self.team1_tricks = 0
        self.team2_tricks = 0

        # Current hand
        self.current_trick: Optional[Trick] = None
        self.tricks_played: List[Trick] = []
        self.hand_number = 0

    def add_player(self, name: str, player_type: PlayerType):
        """Add a player to the game"""
        if len(self.players) >= 4:
            raise ValueError("Game already has 4 players")

        position = len(self.players)
        player = Player(name, player_type, position)
        self.players.append(player)

        if len(self.players) == 4:
            self.phase = GamePhase.DEALING

        return player

    def get_player(self, position: int) -> Player:
        """Get player at a specific position"""
        return self.players[position]

    def get_current_player(self) -> Player:
        """Get the player whose turn it is"""
        return self.players[self.current_player_position]

    def next_player(self):
        """Move to the next player"""
        self.current_player_position = (self.current_player_position + 1) % 4

    def get_team_for_position(self, position: int) -> int:
        """Get team number (1 or 2) for a player position"""
        return 1 if position % 2 == 0 else 2

    def to_dict(
        self, include_hands: bool = False, perspective_position: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Convert game state to dictionary for JSON serialization.

        Args:
            include_hands: Whether to include all player hands
            perspective_position: If set, only show that player's hand
        """
        state = {
            "game_id": self.game_id,
            "phase": self.phase.value,
            "hand_number": self.hand_number,
            "dealer_position": self.dealer_position,
            "current_player_position": self.current_player_position,
            "turned_up_card": str(self.turned_up_card) if self.turned_up_card else None,
            "trump": self.trump.value if self.trump else None,
            "trump_caller_position": self.trump_caller_position,
            "going_alone": self.going_alone,
            "alone_player_position": self.alone_player_position,
            "team1_score": self.team1_score,
            "team2_score": self.team2_score,
            "team1_tricks": self.team1_tricks,
            "team2_tricks": self.team2_tricks,
            "players": [],
            "current_trick": (
                self.current_trick.to_dict() if self.current_trick else None
            ),
            "tricks_played": [t.to_dict() for t in self.tricks_played],
        }

        for player in self.players:
            player_dict = player.to_dict()

            # Add hand information based on parameters
            if include_hands or (
                perspective_position is not None
                and player.position == perspective_position
            ):
                player_dict["hand"] = [str(card) for card in player.hand]

            state["players"].append(player_dict)

        return state


class EuchreGame:
    """Main Euchre game controller"""

    def __init__(self, game_id: str):
        self.state = GameState(game_id)

    def add_player(self, name: str, player_type: PlayerType = PlayerType.HUMAN):
        """Add a player to the game"""
        return self.state.add_player(name, player_type)

    def start_new_hand(self):
        """Start a new hand of Euchre"""
        self.state.hand_number += 1
        self.state.phase = GamePhase.DEALING

        # Reset hand-specific state
        self.state.trump = None
        self.state.trump_caller_position = None
        self.state.turned_up_card = None
        self.state.going_alone = False
        self.state.alone_player_position = None
        self.state.team1_tricks = 0
        self.state.team2_tricks = 0
        self.state.tricks_played = []
        self.state.current_trick = None

        # Clear player hands
        for player in self.state.players:
            player.clear_hand()

        # Deal cards
        self.state.deck.reset()
        self.state.deck.shuffle()

        # Deal 5 cards to each player
        for _ in range(5):
            for i in range(4):
                cards = self.state.deck.deal(1)
                self.state.players[i].add_cards(cards)

        # Turn up a card for trump selection
        self.state.turned_up_card = self.state.deck.draw()

        # Move to trump selection
        self.state.current_player_position = (self.state.dealer_position + 1) % 4
        self.state.phase = GamePhase.TRUMP_SELECTION_ROUND1

    def pass_trump(self) -> bool:
        """
        Current player passes on calling trump.
        Returns True if round advances, False if still in same round.
        """
        if self.state.phase not in [
            GamePhase.TRUMP_SELECTION_ROUND1,
            GamePhase.TRUMP_SELECTION_ROUND2,
        ]:
            raise ValueError("Not in trump selection phase")

        # Check if dealer is forced to call (stick the dealer)
        if self.state.phase == GamePhase.TRUMP_SELECTION_ROUND2:
            if self.state.current_player_position == self.state.dealer_position:
                raise ValueError("Dealer cannot pass in round 2 (stick the dealer)")

        self.state.next_player()

        # Check if we've gone around the table
        next_after_dealer = (self.state.dealer_position + 1) % 4
        if self.state.current_player_position == next_after_dealer:
            if self.state.phase == GamePhase.TRUMP_SELECTION_ROUND1:
                # Move to round 2
                self.state.phase = GamePhase.TRUMP_SELECTION_ROUND2
                return True

        return False

    def call_trump(self, suit: Optional[Suit] = None, go_alone: bool = False):
        """
        Current player calls trump.

        Args:
            suit: Suit to call as trump (None for round 1 = pick up card)
            go_alone: Whether caller is going alone
        """
        if self.state.phase == GamePhase.TRUMP_SELECTION_ROUND1:
            # Calling the turned up card
            if self.state.turned_up_card is None:
                raise ValueError("No turned up card available")
            if suit is not None and suit != self.state.turned_up_card.suit:
                raise ValueError("In round 1, must call turned up suit or pass")

            self.state.trump = self.state.turned_up_card.suit

            # Dealer picks up the card
            dealer = self.state.get_player(self.state.dealer_position)
            dealer.add_cards([self.state.turned_up_card])

            # Move to dealer discard phase
            self.state.trump_caller_position = self.state.current_player_position
            self.state.going_alone = go_alone
            if go_alone:
                self.state.alone_player_position = self.state.current_player_position

            self.state.current_player_position = self.state.dealer_position
            self.state.phase = GamePhase.DEALER_DISCARD

        elif self.state.phase == GamePhase.TRUMP_SELECTION_ROUND2:
            # Can call any suit except the turned up card's suit
            if suit is None:
                raise ValueError("Must specify suit in round 2")
            if self.state.turned_up_card and suit == self.state.turned_up_card.suit:
                raise ValueError("Cannot call turned up suit in round 2")

            self.state.trump = suit
            self.state.trump_caller_position = self.state.current_player_position
            self.state.going_alone = go_alone

            if go_alone:
                self.state.alone_player_position = self.state.current_player_position

            # Start playing (no discard needed in round 2)
            self.state.current_player_position = (self.state.dealer_position + 1) % 4
            self.state.phase = GamePhase.PLAYING
            self._start_new_trick()
        else:
            raise ValueError("Not in trump selection phase")

    def dealer_discard(self, card: Card):
        """
        Dealer discards a card after picking up in round 1.

        Args:
            card: Card to discard from dealer's hand
        """
        if self.state.phase != GamePhase.DEALER_DISCARD:
            raise ValueError("Not in dealer discard phase")

        dealer = self.state.get_player(self.state.dealer_position)

        # Validate card is in dealer's hand
        if not dealer.has_card(card):
            raise ValueError(f"Dealer does not have card {card}")

        # Remove the card
        dealer.remove_card(card)

        # Start playing
        self.state.current_player_position = (self.state.dealer_position + 1) % 4
        self.state.phase = GamePhase.PLAYING
        self._start_new_trick()

    def _start_new_trick(self):
        """Start a new trick"""
        if len(self.state.tricks_played) == 0:
            # First trick - player left of dealer leads
            lead_position = (self.state.dealer_position + 1) % 4
        else:
            # Winner of last trick leads
            lead_position = self.state.tricks_played[-1].get_winner()

        self.state.current_trick = Trick(lead_position, self.state.trump)
        self.state.current_player_position = lead_position

    def play_card(self, card: Card) -> Dict[str, Any]:
        """
        Current player plays a card.
        Returns information about what happened.
        """
        if self.state.phase != GamePhase.PLAYING:
            raise ValueError("Not in playing phase")

        player = self.state.get_current_player()

        # Validate card is in hand
        if not player.has_card(card):
            raise ValueError(f"Player does not have card {card}")

        # Validate card follows suit if required
        lead_suit = self.state.current_trick.lead_suit
        valid_cards = player.get_valid_cards(lead_suit, self.state.trump)
        if card not in valid_cards:
            raise ValueError(f"Must follow suit. Valid cards: {valid_cards}")

        # Play the card
        player.remove_card(card)
        self.state.current_trick.add_card(player.position, card)

        result = {
            "card_played": str(card),
            "player_position": player.position,
            "trick_complete": False,
            "trick_winner": None,
            "hand_complete": False,
            "hand_winner": None,
        }

        # Check if trick is complete
        if self.state.current_trick.is_complete():
            winner_position = self.state.current_trick.get_winner()
            result["trick_complete"] = True
            result["trick_winner"] = winner_position

            # Award trick to team
            team = self.state.get_team_for_position(winner_position)
            if team == 1:
                self.state.team1_tricks += 1
            else:
                self.state.team2_tricks += 1

            self.state.tricks_played.append(self.state.current_trick)

            # Check if hand is complete (5 tricks played)
            if len(self.state.tricks_played) == 5:
                self._complete_hand()
                result["hand_complete"] = True
                result["hand_winner"] = self._determine_hand_winner()
            else:
                # Start next trick
                self._start_new_trick()
        else:
            # Move to next player
            self.state.next_player()

        return result

    def _complete_hand(self):
        """Complete the current hand and award points"""
        self.state.phase = GamePhase.HAND_COMPLETE

    def _determine_hand_winner(self) -> Dict[str, Any]:
        """Determine winner of the hand and award points"""
        team1_tricks = self.state.team1_tricks
        team2_tricks = self.state.team2_tricks

        calling_team = self.state.get_team_for_position(
            self.state.trump_caller_position
        )

        # Determine points
        points = 0
        winning_team = None

        if calling_team == 1:
            if team1_tricks >= 3:
                winning_team = 1
                if team1_tricks == 5:
                    points = 4 if self.state.going_alone else 2  # March
                else:
                    points = 1
            else:
                winning_team = 2
                points = 2  # Euchred
        else:  # calling_team == 2
            if team2_tricks >= 3:
                winning_team = 2
                if team2_tricks == 5:
                    points = 4 if self.state.going_alone else 2  # March
                else:
                    points = 1
            else:
                winning_team = 1
                points = 2  # Euchred

        # Award points
        if winning_team == 1:
            self.state.team1_score += points
        else:
            self.state.team2_score += points

        # Check for game win (usually 10 points)
        if self.state.team1_score >= 10 or self.state.team2_score >= 10:
            self.state.phase = GamePhase.GAME_OVER

        # Move dealer
        self.state.dealer_position = (self.state.dealer_position + 1) % 4

        return {
            "winning_team": winning_team,
            "points_awarded": points,
            "team1_score": self.state.team1_score,
            "team2_score": self.state.team2_score,
            "game_over": self.state.phase == GamePhase.GAME_OVER,
        }

    def get_valid_moves(self, position: Optional[int] = None) -> List[Card]:
        """Get valid moves for a player"""
        if position is None:
            position = self.state.current_player_position

        if self.state.phase != GamePhase.PLAYING:
            return []

        player = self.state.get_player(position)
        lead_suit = (
            self.state.current_trick.lead_suit if self.state.current_trick else None
        )

        return player.get_valid_cards(lead_suit, self.state.trump)

    def get_state(self, perspective_position: Optional[int] = None) -> Dict[str, Any]:
        """Get game state, optionally from a specific player's perspective"""
        return self.state.to_dict(
            include_hands=False, perspective_position=perspective_position
        )
