package com.ubaid.spamshield;

import android.app.Activity;
import android.content.Context;
import android.graphics.Color;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

/** Main screen: editable SMS input -> local TFLite inference -> explainable visual result. */
public final class MainActivity extends Activity {
    private static final String SPAM_SAMPLE =
        "URGENT! You have won a cash prize. Click https://example.com/claim now to collect your reward!";
    private static final String NORMAL_SAMPLE =
        "Hi, the class starts at 9 tomorrow. Please bring your assignment notebook.";

    private EditText messageInput;
    private TextView modelStatus, counter, verdictSymbol, verdictTitle, verdictDescription;
    private TextView spamPercent, normalPercent, featureCount, securityAdvice;
    private ProgressBar spamBar;
    private View resultPanel;
    private Button analyzeButton;
    private SpamClassifier classifier;

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().setStatusBarColor(Color.rgb(20,45,65));
        getWindow().setNavigationBarColor(Color.WHITE);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR);

        messageInput = findViewById(R.id.messageInput);
        modelStatus = findViewById(R.id.modelStatus);
        counter = findViewById(R.id.characterCount);
        verdictSymbol = findViewById(R.id.verdictSymbol);
        verdictTitle = findViewById(R.id.verdictTitle);
        verdictDescription = findViewById(R.id.verdictDescription);
        spamPercent = findViewById(R.id.spamPercent);
        normalPercent = findViewById(R.id.normalPercent);
        featureCount = findViewById(R.id.featureCount);
        securityAdvice = findViewById(R.id.securityAdvice);
        spamBar = findViewById(R.id.spamBar);
        resultPanel = findViewById(R.id.resultPanel);
        analyzeButton = findViewById(R.id.analyzeButton);

        analyzeButton.setEnabled(false);
        try {
            classifier = new SpamClassifier(this);
            modelStatus.setText("●  AI model ready  ·  All analysis stays on this phone");
            analyzeButton.setEnabled(true);
        } catch (Exception e) {
            modelStatus.setText("Model unavailable — check spamshield.tflite and vocabulary.txt");
            Toast.makeText(this, "AI model load failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }

        messageInput.addTextChangedListener(new TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int start, int count, int after) { }
            @Override public void onTextChanged(CharSequence s, int start, int before, int count) {
                counter.setText(s.length() + " / 500");
                // If user edits the text, never show an old result as if it were current.
                resultPanel.setVisibility(View.GONE);
            }
            @Override public void afterTextChanged(Editable s) { }
        });
        findViewById(R.id.spamExample).setOnClickListener(v -> useExample(SPAM_SAMPLE));
        findViewById(R.id.normalExample).setOnClickListener(v -> useExample(NORMAL_SAMPLE));
        findViewById(R.id.clearButton).setOnClickListener(v -> {
            messageInput.setText("");
            messageInput.clearFocus();
            resultPanel.setVisibility(View.GONE);
            hideKeyboard();
        });
        analyzeButton.setOnClickListener(v -> analyze());
    }

    private void useExample(String value) {
        messageInput.setText(value);
        messageInput.setSelection(messageInput.length());
        messageInput.clearFocus();
        hideKeyboard();
    }

    private void analyze() {
        String message = messageInput.getText().toString().trim();
        if (message.isEmpty()) {
            messageInput.setError("Paste or enter a message to analyze");
            messageInput.requestFocus();
            return;
        }
        if (classifier == null) {
            Toast.makeText(this, "AI model unavailable", Toast.LENGTH_SHORT).show();
            return;
        }
        hideKeyboard();
        try {
            SpamClassifier.Result result = classifier.predict(message);
            showResult(result);
        } catch (IllegalArgumentException e) {
            messageInput.setError(e.getMessage());
        } catch (Exception e) {
            Toast.makeText(this, "Could not analyze: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void showResult(SpamClassifier.Result result) {
        float risk = result.spamScore;
        int percent = Math.round(risk * 100f);
        spamBar.setProgress(percent);
        spamPercent.setText(percent + "%");
        normalPercent.setText("Normal-message score: " + Math.round(result.normalScore * 100f) + "%");
        featureCount.setText("Recognized " + result.matchedFeatures + " model features from this message.");

        // The three UI bands are display guidance; the ML score always comes from the model.
        if (risk >= 0.72f) {
            verdictSymbol.setText("⚠");
            verdictSymbol.setTextColor(Color.rgb(206,76,88));
            verdictTitle.setText("Likely spam");
            verdictDescription.setText("This message resembles promotional or fraudulent messages in the training examples.");
            securityAdvice.setText("Safety tip: Don't open unexpected links, send money, or share a password or OTP. Verify any request through the official app or phone number.");
        } else if (risk >= 0.35f) {
            verdictSymbol.setText("?");
            verdictSymbol.setTextColor(Color.rgb(180,120,22));
            verdictTitle.setText("Review carefully");
            verdictDescription.setText("The model is uncertain. Don't treat this score as a guarantee that the message is safe.");
            securityAdvice.setText("Safety tip: Be cautious of urgency, payment requests, unfamiliar links, and requests for personal information. Check with the sender independently.");
        } else {
            verdictSymbol.setText("✓");
            verdictSymbol.setTextColor(Color.rgb(22,139,107));
            verdictTitle.setText("Likely normal");
            verdictDescription.setText("The model found fewer patterns associated with spam in its educational training data.");
            securityAdvice.setText("Reminder: This is an AI estimate, not a security guarantee. Unexpected links and requests for passwords or OTPs always deserve independent verification.");
        }
        resultPanel.setVisibility(View.VISIBLE);
        View root = findViewById(R.id.rootScroll);
        root.post(() -> resultPanel.requestFocus());
    }

    private void hideKeyboard() {
        InputMethodManager imm = (InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
        if (imm != null) imm.hideSoftInputFromWindow(messageInput.getWindowToken(),0);
    }

    @Override protected void onDestroy() {
        if (classifier != null) {
            classifier.close();
            classifier = null;
        }
        super.onDestroy();
    }
}
