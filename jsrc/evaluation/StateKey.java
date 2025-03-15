package evaluation;

import java.util.Arrays;

public final class StateKey {
  private final int[] valueCounts;
  private final int playerScore;
  private final int dealerScore;
  private final boolean playerSoft;
  private final boolean isSplit;
  private final String action;
  private final int hash;

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

  @Override
  public int hashCode() {
    return hash;
  }
}
