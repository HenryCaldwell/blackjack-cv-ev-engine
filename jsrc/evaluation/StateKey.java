package evaluation;

import java.util.Arrays;

/**
 * Represents a unique state key for a blackjack game configuration. The state
 * key includes the deck's card distribution, the player's and dealer's scores,
 * whether the player's hand is soft, whether the hand is a result of a split,
 * and the action being considered. The key is immutable and pre-computes its
 * hash code for performance.
 */
public final class StateKey {
  private final int[] valueCounts;
  private final int playerScore;
  private final int dealerScore;
  private final boolean playerSoft;
  private final boolean isSplit;
  private final String action;
  private final int hash;

  /**
   * Constructs a StateKey with the specified game state parameters.
   *
   * @param valueCounts an array representing the current distribution of card
   *                    values in the deck.
   * @param playerScore the total score of the player's hand.
   * @param dealerScore the total score of the dealer's hand.
   * @param playerSoft  {@code true} if the player's hand is soft; {@code false}
   *                    otherwise.
   * @param isSplit     {@code true} if the hand is a result of a split;
   *                    {@code false} otherwise.
   * @param action      the action being considered (e.g., "stand", "hit",
   *                    "double", or "split").
   */
  public StateKey(int[] valueCounts, int playerScore, int dealerScore, boolean playerSoft, boolean isSplit,
      String action) {
    this.valueCounts = Arrays.copyOf(valueCounts, valueCounts.length);
    this.playerScore = playerScore;
    this.dealerScore = dealerScore;
    this.playerSoft = playerSoft;
    this.isSplit = isSplit;
    this.action = action;

    int h = Arrays.hashCode(this.valueCounts);
    h = 31 * h + playerScore;
    h = 31 * h + dealerScore;
    h = 31 * h + Boolean.hashCode(playerSoft);
    h = 31 * h + Boolean.hashCode(isSplit);
    h = 31 * h + action.hashCode();
    this.hash = h;
  }

  /**
   * Compares this StateKey with the specified object for equality.
   *
   * Two StateKey objects are considered equal if they have identical card
   * distributions, player scores, dealer scores, soft hand flags, split flags,
   * and actions.
   *
   * @param o the object to be compared for equality with this StateKey.
   * @return {@code true} if the specified object is equal to this StateKey;
   *         {@code false} otherwise.
   */
  @Override
  public boolean equals(Object o) {
    if (this == o)
      return true;

    if (!(o instanceof StateKey))
      return false;

    StateKey key = (StateKey) o;

    return playerScore == key.playerScore &&
        dealerScore == key.dealerScore &&
        playerSoft == key.playerSoft &&
        isSplit == key.isSplit &&
        Arrays.equals(valueCounts, key.valueCounts) &&
        action.equals(key.action);
  }

  /**
   * Returns the pre-computed hash code for this StateKey.
   *
   * @return the hash code.
   */
  @Override
  public int hashCode() {
    return hash;
  }
}
