package evaluation;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Represents an engine for calculating the expected values (EV) of various
 * actions in a blackjack game. The EV represents the average outcome of a
 * decision over many iterations. This class uses memoization to cache results
 * and optimize recursive EV calculations.
 */
public class EVCalculator {
  private Map<StateKey, Double> cache;

  /**
   * Constructs an EVCalculator instance and initializes the cache.
   */
  public EVCalculator() {
    this.cache = new HashMap<>();
  }

  // ------------------------------------------------------------------------
  // Public API Methods
  // ------------------------------------------------------------------------

  /**
   * Calculates the expected value when the player chooses to stand.
   *
   * @param valueCounts An array representing the current distribution of card
   *                    values in the deck.
   * @param playerHand  A list of integers representing the player's current hand.
   * @param dealerHand  A list of integers representing the dealer's current hand.
   * @return The expected value (EV) for standing.
   * @throws IllegalArgumentException if any of the arguments are {@code null}.
   */
  public double calculateStandEV(int[] valueCounts, List<Integer> playerHand, List<Integer> dealerHand) {
    if (valueCounts == null || playerHand == null || dealerHand == null) {
      throw new IllegalArgumentException(
          "Arguments to calculateStandEV cannot be null: valueCounts, playerHand, and dealerHand are required");
    }

    return calculateStandEV(valueCounts, playerHand, dealerHand, false);
  }

  /**
   * Calculates the expected value when the player chooses to hit.
   *
   * @param valueCounts An array representing the current distribution of card
   *                    values in the deck.
   * @param playerHand  A list of integers representing the player's current hand.
   * @param dealerHand  A list of integers representing the dealer's current hand.
   * @return The expected value (EV) for hitting.
   * @throws IllegalArgumentException if any of the arguments are {@code null}.
   */
  public double calculateHitEV(int[] valueCounts, List<Integer> playerHand, List<Integer> dealerHand) {
    if (valueCounts == null || playerHand == null || dealerHand == null) {
      throw new IllegalArgumentException(
          "Arguments to calculateHitEV cannot be null: valueCounts, playerHand, and dealerHand are required");
    }

    return calculateHitEV(valueCounts, playerHand, dealerHand, false);
  }

  /**
   * Calculates the expected value when the player chooses to double.
   *
   * @param valueCounts An array representing the current distribution of card
   *                    values in the deck.
   * @param playerHand  A list of integers representing the player's current hand.
   * @param dealerHand  A list of integers representing the dealer's current hand.
   * @return The expected value (EV) for doubling.
   * @throws IllegalArgumentException if any of the arguments are {@code null}.
   */
  public double calculateDoubleEV(int[] valueCounts, List<Integer> playerHand, List<Integer> dealerHand) {
    if (valueCounts == null || playerHand == null || dealerHand == null) {
      throw new IllegalArgumentException(
          "Arguments to calculateDoubleEV cannot be null: valueCounts, playerHand, and dealerHand are required");
    }

    return calculateDoubleEV(valueCounts, playerHand, dealerHand, false);
  }

  /**
   * Calculates the expected value when the player chooses to split.
   *
   * @param valueCounts An array representing the current distribution of card
   *                    values in the deck.
   * @param playerHand  A list of integers representing the player's current hand.
   * @param dealerHand  A list of integers representing the dealer's current hand.
   * @return The expected value (EV) for splitting.
   * @throws IllegalArgumentException if any of the arguments are {@code null} or
   *                                  if the player's hand cannot be
   *                                  split.
   */
  public double calculateSplitEV(int[] valueCounts, List<Integer> playerHand, List<Integer> dealerHand) {
    if (valueCounts == null || playerHand == null || dealerHand == null) {
      throw new IllegalArgumentException(
          "Arguments to calculateSplitEV cannot be null: valueCounts, playerHand, and dealerHand are required");
    }

    if (!canSplitHand(playerHand)) {
      return Double.NEGATIVE_INFINITY;
    }

    return calculateSplitEV(valueCounts, playerHand, dealerHand, true);
  }

  // ------------------------------------------------------------------------
  // Private Recursive Calculation Methods
  // ------------------------------------------------------------------------

  /**
   * Recursively computes the expected value for standing based on the current
   * card distribution and both the player's and dealer's hands.
   *
   * @param valueCounts An array representing the current distribution of card
   *                    values in the deck.
   * @param playerHand  A list of integers representing the player's hand.
   * @param dealerHand  A list of integers representing the dealer's hand.
   * @param isSplit     Indicates whether the hand is a result of a split.
   * @return The expected value (EV) for standing.
   */
  private double calculateStandEV(int[] valueCounts, List<Integer> playerHand, List<Integer> dealerHand,
      boolean isSplit) {
    StateKey stateKey = getStateKey(valueCounts, playerHand, dealerHand, isSplit, "stand");

    if (cache.containsKey(stateKey)) {
      return cache.get(stateKey);
    }

    int dealerScore = calculateHandScore(dealerHand);
    boolean isSoft = isSoftHand(dealerHand);

    if (dealerScore > 17 || (dealerScore == 17 && (!isSoft || (isSoft && !ConfigManager.DEALER_HITS_ON_SOFT_17)))) {
      double outcome = evaluateOutcome(playerHand, dealerHand, isSplit);
      cache.put(stateKey, outcome);

      return outcome;
    }

    double totalValue = 0.0;
    int totalCards = 0;

    for (int i = 0; i < valueCounts.length; i++) {
      if (valueCounts[i] > 0) {
        if (ConfigManager.DEALER_PEAKS_FOR_21 && dealerHand.size() == 1 &&
            ((dealerHand.get(0) == 10 && i == 0) || (dealerHand.get(0) == 1 && i == 9))) {
          continue;
        }

        int count = valueCounts[i];
        valueCounts[i]--;

        int newCard = (i == 0) ? 1 : (i < 9 ? i + 1 : 10);
        dealerHand.add(newCard);

        double outcome = calculateStandEV(valueCounts, playerHand, dealerHand, isSplit);
        totalValue += outcome * count;
        totalCards += count;

        dealerHand.remove(dealerHand.size() - 1);
        valueCounts[i]++;
      }
    }

    double EV = (totalCards > 0) ? totalValue / totalCards : 0.0;
    cache.put(stateKey, EV);

    return EV;
  }

  /**
   * Recursively computes the expected value for hitting based on the current card
   * distribution and both the player's and dealer's hands.
   *
   * @param valueCounts An array representing the current distribution of card
   *                    values in the deck.
   * @param playerHand  A list of integers representing the player's hand.
   * @param dealerHand  A list of integers representing the dealer's hand.
   * @param isSplit     Indicates whether the hand is a result of a split.
   * @return The expected value (EV) for hitting.
   */
  private double calculateHitEV(int[] valueCounts, List<Integer> playerHand, List<Integer> dealerHand,
      boolean isSplit) {
    StateKey stateKey = getStateKey(valueCounts, playerHand, dealerHand, isSplit, "hit");

    if (cache.containsKey(stateKey)) {
      return cache.get(stateKey);
    }

    double totalValue = 0.0;
    int totalCards = 0;

    for (int i = 0; i < valueCounts.length; i++) {
      if (valueCounts[i] > 0) {
        int count = valueCounts[i];
        valueCounts[i]--;

        int newCard = (i == 0) ? 1 : (i < 9 ? i + 1 : 10);
        playerHand.add(newCard);

        if (calculateHandScore(playerHand) > 21) {
          totalValue -= count;
          totalCards += count;
        } else {
          double standEV = calculateStandEV(valueCounts, playerHand, dealerHand, isSplit);
          double hitEV = calculateHitEV(valueCounts, playerHand, dealerHand, isSplit);
          double maxEV = Math.max(standEV, hitEV);
          totalValue += maxEV * count;
          totalCards += count;
        }

        playerHand.remove(playerHand.size() - 1);
        valueCounts[i]++;
      }
    }

    double EV = (totalCards > 0) ? totalValue / totalCards : 0.0;
    cache.put(stateKey, EV);
    return EV;
  }

  /**
   * Recursively computes the expected value for doubling based on the current
   * card distribution and both the player's and dealer's hands.
   *
   * @param valueCounts An array representing the current distribution of card
   *                    values in the deck.
   * @param playerHand  A list of integers representing the player's hand.
   * @param dealerHand  A list of integers representing the dealer's hand.
   * @param isSplit     Indicates whether the hand is a result of a split.
   * @return The expected value (EV) for doubling.
   */
  private double calculateDoubleEV(int[] valueCounts, List<Integer> playerHand, List<Integer> dealerHand,
      boolean isSplit) {
    StateKey stateKey = getStateKey(valueCounts, playerHand, dealerHand, isSplit, "double");

    if (cache.containsKey(stateKey)) {
      return cache.get(stateKey);
    }

    double totalValue = 0.0;
    int totalCards = 0;

    for (int i = 0; i < valueCounts.length; i++) {
      if (valueCounts[i] > 0) {
        int count = valueCounts[i];
        valueCounts[i]--;

        int newCard = (i == 0) ? 1 : (i < 9 ? i + 1 : 10);
        playerHand.add(newCard);

        if (calculateHandScore(playerHand) > 21) {
          totalValue -= 2.0 * count;
          totalCards += count;
        } else {
          double outcome = 2.0 * calculateStandEV(valueCounts, playerHand, dealerHand, isSplit);
          totalValue += outcome * count;
          totalCards += count;
        }

        playerHand.remove(playerHand.size() - 1);
        valueCounts[i]++;
      }
    }

    double EV = (totalCards > 0) ? totalValue / totalCards : 0.0;
    cache.put(stateKey, EV);

    return EV;
  }

  /**
   * Recursively computes the expected value for splitting based on the current
   * card distribution and both the player's and dealer's hands.
   *
   * @param valueCounts An array representing the current distribution of card
   *                    values in the deck.
   * @param playerHand  A list of integers representing the player's hand.
   * @param dealerHand  A list of integers representing the dealer's hand.
   * @param isSplit     This parameter is always {@code true} for split
   *                    calculations.
   * @return The expected value (EV) for splitting.
   */
  private double calculateSplitEV(int[] valueCounts, List<Integer> playerHand, List<Integer> dealerHand,
      boolean isSplit) {
    StateKey stateKey = getStateKey(valueCounts, playerHand, dealerHand, true, "split");

    if (cache.containsKey(stateKey)) {
      return cache.get(stateKey);
    }

    int splitCard = playerHand.get(0);
    boolean isAceSplit = splitCard == 1;

    double totalValue = 0.0;
    int totalCards = 0;

    int secondCard = playerHand.get(1);
    playerHand.remove(1);

    for (int i = 0; i < valueCounts.length; i++) {
      if (valueCounts[i] > 0) {
        int count = valueCounts[i];
        valueCounts[i]--;

        int newCard = (i == 0) ? 1 : (i < 9 ? i + 1 : 10);
        playerHand.add(newCard);

        double standEV = calculateStandEV(valueCounts, playerHand, dealerHand, true);
        double hitEV = Double.NEGATIVE_INFINITY;
        double doubleEV = Double.NEGATIVE_INFINITY;

        if (isAceSplit && ConfigManager.HIT_SPLIT_ACES || !isAceSplit) {
          hitEV = calculateHitEV(valueCounts, playerHand, dealerHand, true);
        }

        if (ConfigManager.DOUBLE_AFTER_SPLIT
            && ((isAceSplit && ConfigManager.HIT_SPLIT_ACES && ConfigManager.DOUBLE_SPLIT_ACES) || !isAceSplit)) {
          doubleEV = calculateDoubleEV(valueCounts, playerHand, dealerHand, true);
        }

        double outcome = Math.max(standEV, Math.max(hitEV, doubleEV));
        totalValue += 2 * outcome * count;
        totalCards += count;

        playerHand.remove(playerHand.size() - 1);
        valueCounts[i]++;
      }
    }

    double EV = (totalCards > 0) ? totalValue / totalCards : 0.0;
    cache.put(stateKey, EV);

    playerHand.add(secondCard);

    return EV;
  }

  // ------------------------------------------------------------------------
  // Helper Methods
  // ------------------------------------------------------------------------

  /**
   * Calculates the total score of a hand. Aces are counted as either 1 or 11 to
   * maximize the hand's value without busting.
   *
   * @param cards A list of integers representing the cards in the hand.
   * @return The total score of the hand.
   */
  private int calculateHandScore(List<Integer> cards) {
    int totalScore = 0;
    int aceCount = 0;

    for (int card : cards) {
      if (card == 1) {
        totalScore++;
        aceCount++;
      } else {
        totalScore += card;
      }
    }

    while (aceCount > 0 && totalScore + 10 <= 21) {
      totalScore += 10;
      aceCount--;
    }

    return totalScore;
  }

  /**
   * Determines if a hand is "soft", meaning it contains an ace counted as 11
   * without busting.
   *
   * @param cards A list of integers representing the cards in the hand.
   * @return {@code true} if the hand is soft; {@code false} otherwise.
   */
  private boolean isSoftHand(List<Integer> cards) {
    int totalScore = 0;
    int aceCount = 0;

    for (int card : cards) {
      if (card == 1) {
        totalScore++;
        aceCount++;
      } else {
        totalScore += card;
      }
    }

    return aceCount > 0 && totalScore + 10 <= 21;
  }

  /**
   * Checks whether the player's hand can be split. A hand can be split if it
   * consists of exactly two cards and both cards are equal, or if both cards are
   * tens.
   *
   * @param cards A list of integers representing the player's hand.
   * @return {@code true} if the hand can be split; {@code false} otherwise.
   */
  private boolean canSplitHand(List<Integer> cards) {
    if (cards.size() != 2) {
      return false;
    }

    int card1 = cards.get(0);
    int card2 = cards.get(1);

    if (card1 == 10 && card2 == 10) {
      return true;
    }

    return card1 == card2;
  }

  /**
   * Evaluates the final outcome of a hand based on the player's and dealer's
   * scores. Positive values indicate a win, negative values indicate a loss, and
   * zero represents a push.
   *
   * @param playerHand A list of integers representing the player's hand.
   * @param dealerHand A list of integers representing the dealer's hand.
   * @param isSplit    Indicates if the hand is a result of a split.
   * @return The numeric outcome of the hand.
   */
  private double evaluateOutcome(List<Integer> playerHand, List<Integer> dealerHand, boolean isSplit) {
    int playerScore = calculateHandScore(playerHand);
    int dealerScore = calculateHandScore(dealerHand);
    int playerHandSize = playerHand.size();
    int dealerHandSize = dealerHand.size();

    boolean playerNaturalBlackjack = playerScore == 21 && playerHandSize == 2
        && (!isSplit || ConfigManager.NATURAL_BLACKJACK_SPLITS);
    boolean dealerNaturalBlackjack = dealerScore == 21 && dealerHandSize == 2;

    if (playerNaturalBlackjack && dealerNaturalBlackjack) {
      return 0.0;
    } else if (playerNaturalBlackjack) {
      return ConfigManager.BLACKJACK_ODDS;
    } else if (dealerNaturalBlackjack) {
      return -1.0;
    } else if (playerScore > 21) {
      return -1.0;
    } else if (dealerScore > 21) {
      return 1.0;
    } else if (playerScore > dealerScore) {
      return 1.0;
    } else if (playerScore < dealerScore) {
      return -1.0;
    } else {
      return 0.0;
    }
  }

  // ------------------------------------------------------------------------
  // State Key Utilities
  // ------------------------------------------------------------------------

  /**
   * Generates a unique state key for the current game configuration. This key is
   * used for memoization to avoid redundant calculations.
   *
   * @param valueCounts An array representing the current distribution of card
   *                    values in the deck.
   * @param playerHand  A list of integers representing the player's hand.
   * @param dealerHand  A list of integers representing the dealer's hand.
   * @param isSplit     Indicates whether the hand is a result of a split.
   * @param action      The action being considered ("stand", "hit", "double", or
   *                    "split").
   * @return A {@code StateKey} that uniquely identifies the current game state.
   */
  private StateKey getStateKey(int[] valueCounts, List<Integer> playerHand, List<Integer> dealerHand, boolean isSplit,
      String action) {
    int playerScore = calculateHandScore(playerHand);
    int dealerScore = calculateHandScore(dealerHand);
    boolean playerSoft = isSoftHand(playerHand);

    return new StateKey(valueCounts, playerScore, dealerScore, playerSoft, isSplit, action);
  }
}