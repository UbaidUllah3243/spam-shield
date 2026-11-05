package com.ubaid.spamshield;

import org.junit.Test;
import java.util.Arrays;
import static org.junit.Assert.*;

public class MessageVectorizerTest {
    @Test public void urlAndPhoneNormalization() {
        MessageVectorizer vectorizer = new MessageVectorizer(
            Arrays.asList("claim", "urltoken", "numbertoken", "claim urltoken", "now"));
        MessageVectorizer.FeatureVector v = vectorizer.vectorize(
            "Claim https://example.com NOW 12345678");
        assertArrayEquals(new float[]{1,1,1,1,1}, v.values, 0.0001f);
        assertEquals(5,v.matched);
    }
    @Test public void repeatedTokensAreBinary() {
        MessageVectorizer vectorizer = new MessageVectorizer(Arrays.asList("free", "free free"));
        MessageVectorizer.FeatureVector v = vectorizer.vectorize("FREE free free");
        assertArrayEquals(new float[]{1,1},v.values,0.0001f);
    }
    @Test public void unmatchedInputReturnsEmptyVector() {
        MessageVectorizer vectorizer = new MessageVectorizer(Arrays.asList("claim"));
        assertEquals(0,vectorizer.vectorize("zzzzzyyyy").matched);
    }
}
