package com.ubaid.spamshield;

import android.content.Context;
import android.content.res.AssetManager;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.DataType;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * A real LiteRT / TensorFlow Lite on-device inference pipeline.
 *
 * Input: [1, N] FLOAT32 binary word/bigram features from MessageVectorizer.
 * Output: [1, 2] FLOAT32 softmax scores, order: NORMAL (0), SPAM (1).
 * Running this does NOT contact a web API or read the user's SMS inbox.
 */
public final class SpamClassifier implements AutoCloseable {
    private final Interpreter interpreter;
    private final MessageVectorizer vectorizer;

    public SpamClassifier(Context context) throws IOException {
        AssetManager assets = context.getAssets();
        List<String> vocabulary = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                assets.open("vocabulary.txt"), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) vocabulary.add(line);
        }
        vectorizer = new MessageVectorizer(vocabulary);

        // Load the model from assets; direct byte buffers work even for compressed assets.
        ByteBuffer model;
        try (InputStream input = assets.open("spamshield.tflite");
             ByteArrayOutputStream data = new ByteArrayOutputStream()) {
            byte[] block = new byte[8192];
            int count;
            while ((count = input.read(block)) != -1) data.write(block, 0, count);
            byte[] bytes = data.toByteArray();
            model = ByteBuffer.allocateDirect(bytes.length).order(ByteOrder.nativeOrder());
            model.put(bytes);
            model.rewind();
        }
        interpreter = new Interpreter(model, new Interpreter.Options().setNumThreads(2));
        int[] inputShape = interpreter.getInputTensor(0).shape();
        int[] outputShape = interpreter.getOutputTensor(0).shape();
        if (inputShape.length != 2 || inputShape[0] != 1 || inputShape[1] != vectorizer.size()
                || outputShape.length != 2 || outputShape[0] != 1 || outputShape[1] != 2
                || interpreter.getInputTensor(0).dataType() != DataType.FLOAT32
                || interpreter.getOutputTensor(0).dataType() != DataType.FLOAT32) {
            interpreter.close();
            throw new IOException("Model does not match the packaged vocabulary");
        }
    }

    public Result predict(String text) {
        MessageVectorizer.FeatureVector vector = vectorizer.vectorize(text);
        if (vector.matched == 0) {
            throw new IllegalArgumentException("No recognizable English words. Please enter a message.");
        }
        float[][] input = new float[1][vectorizer.size()];
        System.arraycopy(vector.values, 0, input[0], 0, vector.values.length);
        float[][] output = new float[1][2];

        // THIS is real AI inference: the TFLite model computes the output scores.
        interpreter.run(input, output);
        float spam = output[0][1];
        if (Float.isNaN(spam) || Float.isInfinite(spam) || spam < 0f || spam > 1f) {
            throw new IllegalStateException("The model produced an invalid score");
        }
        return new Result(output[0][0], spam, vector.matched);
    }

    @Override public void close() { interpreter.close(); }

    public static final class Result {
        public final float normalScore;
        public final float spamScore;
        public final int matchedFeatures;
        Result(float normal, float spam, int count) {
            normalScore = normal; spamScore = spam; matchedFeatures = count;
        }
    }
}
