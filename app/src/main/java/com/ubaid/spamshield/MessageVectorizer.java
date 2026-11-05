package com.ubaid.spamshield;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Recreates the exact Python model input preparation.
 * 1) Replace URLs and long numeric sequences with stable feature tokens.
 * 2) Find lowercase English words including contractions.
 * 3) Construct a binary bag of single words + adjacent pairs (bigrams).
 *
 * This class does NOT label messages with hand-written spam rules. Only the
 * trained model in app/src/main/assets/spamshield.tflite does that.
 */
public final class MessageVectorizer {
    private static final Pattern URL = Pattern.compile("(?i)(?:https?://|www\\.)\\S+");
    private static final Pattern LONG_NUMBER = Pattern.compile("\\b[0-9]{5,}\\b");
    private static final Pattern WORD = Pattern.compile("[a-z]+(?:'[a-z]+)?");
    private final Map<String, Integer> features = new HashMap<>();

    public MessageVectorizer(List<String> vocabulary) {
        int position = 0;
        for (String entry : vocabulary) {
            String key = entry.trim();
            if (key.isEmpty()) continue;
            if (features.containsKey(key)) throw new IllegalArgumentException("Duplicate feature: " + key);
            features.put(key, position++);
        }
        if (features.isEmpty()) throw new IllegalArgumentException("Empty vocabulary");
    }

    public int size() { return features.size(); }

    public FeatureVector vectorize(String input) {
        float[] vector = new float[features.size()];
        if (input == null || input.trim().isEmpty()) return new FeatureVector(vector, 0);
        String normalized = input.toLowerCase(Locale.ROOT);
        normalized = URL.matcher(normalized).replaceAll(" urltoken ");
        normalized = LONG_NUMBER.matcher(normalized).replaceAll(" numbertoken ");
        normalized = normalized.replace("£", " moneytoken ").replace("$", " moneytoken ")
                               .replace("€", " moneytoken ");
        Matcher m = WORD.matcher(normalized);
        List<String> tokens = new ArrayList<>();
        while (m.find()) tokens.add(m.group());
        int matched = 0;
        for (int i = 0; i < tokens.size(); i++) {
            matched += set(tokens.get(i), vector);
            if (i + 1 < tokens.size()) matched += set(tokens.get(i) + " " + tokens.get(i + 1), vector);
        }
        return new FeatureVector(vector, matched);
    }

    private int set(String feature, float[] vector) {
        Integer index = features.get(feature);
        if (index == null || vector[index] != 0f) return 0;
        vector[index] = 1f;
        return 1;
    }

    public static final class FeatureVector {
        public final float[] values;
        public final int matched;
        private FeatureVector(float[] v, int n) { values = v; matched = n; }
    }
}
