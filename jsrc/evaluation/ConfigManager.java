package evaluation;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;

import org.yaml.snakeyaml.Yaml;

/**
 * Represents a configuration manager for the blackjack evaluation engine. The
 * settings include odds, dealer behavior, and options for splitting, doubling,
 * and surrendering. If the configuration file cannot be loaded or parsed, an
 * ExceptionInInitializerError is thrown. This class is non-instantiable.
 */
public final class ConfigManager {
  public static final double BLACKJACK_ODDS;
  public static final boolean DEALER_HITS_ON_SOFT_17;
  public static final boolean DEALER_PEAKS_FOR_21;
  public static final boolean NATURAL_BLACKJACK_SPLITS;
  public static final boolean DOUBLE_AFTER_SPLIT;
  public static final boolean HIT_SPLIT_ACES;
  public static final boolean DOUBLE_SPLIT_ACES;

  // Static block to load configuration from config.yaml at class initialization
  static {
    Yaml yaml = new Yaml();

    try (InputStream in = new FileInputStream("config.yaml")) {
      Map<String, Object> config = yaml.load(in);
      @SuppressWarnings("unchecked")
      Map<String, Object> gameSettings = (Map<String, Object>) config.get("analysis_settings");

      BLACKJACK_ODDS = ((Number) gameSettings.get("blackjack_odds")).doubleValue();

      DEALER_HITS_ON_SOFT_17 = (Boolean) gameSettings.get("dealer_hits_on_soft_17");
      DEALER_PEAKS_FOR_21 = (Boolean) gameSettings.get("dealer_peaks_for_21");

      NATURAL_BLACKJACK_SPLITS = (Boolean) gameSettings.get("natural_blackjack_splits");
      DOUBLE_AFTER_SPLIT = (Boolean) gameSettings.get("double_after_split");
      HIT_SPLIT_ACES = (Boolean) gameSettings.get("hit_split_aces");
      DOUBLE_SPLIT_ACES = (Boolean) gameSettings.get("double_split_aces");
    } catch (IOException e) {
      throw new ExceptionInInitializerError("Failed to load config.yaml: " + e.getMessage());
    } catch (Exception e) {
      throw new ExceptionInInitializerError("Failed to parse config.yaml: " + e.getMessage());
    }
  }

  /**
   * Private constructor to prevent instantiation.
   */
  private ConfigManager() {
    throw new UnsupportedOperationException("GameSettings is a utility class and cannot be instantiated");
  }
}