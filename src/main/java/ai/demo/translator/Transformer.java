package ai.demo.translator;

import java.util.*;
import static ai.demo.translator.App.OUT;
import static ai.demo.translator.ParameterReader.*;
import static ai.demo.translator.TransformerUtil.*;

/**
 * Encoder-decoder transformer implementation
 */
public class Transformer
{
    private final Settings settings;
    private final Tokenizer tokenizer;

    private final float[][] tokenEmbeddings;
    private final float[][] encoderPositionEmbeddings;
    private final float[] encoderNormFinalWeights;
    private final float[] encoderNormFinalBiases;

    private final float[][] decoderPositionEmbeddings;
    private final float[] decoderNormFinalWeights;
    private final float[] decoderNormFinalBiases;

    private final TransformerEncoder[] encoders;
    private final TransformerDecoder[] decoders;

    /**
     * Initialization
     */
    public Transformer(Settings settings, Tokenizer tokenizer)
    {
        String path = settings.getPath();
        int hiddenSize = settings.getHiddenSize();

        this.settings = settings;
        this.tokenizer = tokenizer;

        this.tokenEmbeddings = readMatrixFile(path, "encoders/input/wte", settings.getTokenCount(), hiddenSize);
        this.encoderPositionEmbeddings = readMatrixFile(path, "encoders/input/wpe", settings.getContextSize(), hiddenSize);
        this.encoderNormFinalWeights = readVectorFile(path, "encoders/output/norm.w", hiddenSize);
        this.encoderNormFinalBiases = readVectorFile(path, "encoders/output/norm.b", hiddenSize);

        this.decoderPositionEmbeddings = readMatrixFile(path, "decoders/input/wpe", settings.getContextSize(), hiddenSize);
        this.decoderNormFinalWeights = readVectorFile(path, "decoders/output/norm.w", hiddenSize);
        this.decoderNormFinalBiases = readVectorFile(path, "decoders/output/norm.b", hiddenSize);

        this.encoders = new TransformerEncoder[settings.getEncoderCount()];
        for (int i = 0; i < settings.getEncoderCount(); i++)
        {
            this.encoders[i] = new TransformerEncoder(i, settings);
        }

        this.decoders = new TransformerDecoder[settings.getDecoderCount()];
        for (int i = 0; i < settings.getDecoderCount(); i++)
        {
            this.decoders[i] = new TransformerDecoder(i, settings);
        }
    }

    /**
     * Transformer token processing logic
     * This method implements the logic how the input tokens and the new and new generated tokens are passed to the transformer
     */
    public List<Integer> processTokens(List<Integer> inputTokens)
    {
        // Adding the START-OF-TEXT token to the beginning.
        // The output for this token will be the output of the whole encoder stack
        // (The output of the other tokens will be dropped, these will be used only by the attention mechanism.)
        int startToken = settings.getStartOfTextToken();
        inputTokens.add(0, startToken);

        float[] encoderOutput = encoderStack(inputTokens);

        // Collector of the generated new tokens (translation)
        List<Integer> result = new ArrayList<>();

        float[] hiddenState = tokenEmbeddings[startToken];

        for (int pos = 0; pos < settings.getContextSize(); pos++)
        {
            // Add the last input token or the previously generated new token as input
            hiddenState = decoderStack(pos, hiddenState, encoderOutput);

            // The output will be the next new token
            int nextToken = selectNextToken(hiddenState);
            result.add(nextToken);

            // Exit if the END_OF_TEXT token was chosen or the context size is reached
            if (nextToken == settings.getEndOfTextToken()) break;
        }

        return result;
    }

    private float[] encoderStack(List<Integer> inputTokens)
    {
        // TODO

        // Word token embedding
        float[] hiddenState = tokenEmbeddings[token];

        // Position embedding
        hiddenState = Util.addVectors(hiddenState, encoderPositionEmbeddings[pos]);

        // Encoder stack
        for (TransformerEncoder encoder : encoders)
        {
            hiddenState = encoder.execute(hiddenState);
        }

        // Final normalization
        return normalization(hiddenState, encoderNormFinalWeights, encoderNormFinalBiases, settings.getEpsilon());
    }

    private float[] decoderStack(int pos, float[] hiddenState, float[] encoderOutput)
    {
        // Position embedding
        hiddenState = Util.addVectors(hiddenState, decoderPositionEmbeddings[pos]);

        // Decoder stack
        for (TransformerDecoder decoder : decoders)
        {
            hiddenState = decoder.execute(hiddenState, encoderOutput);
        }

        // Final normalization
        return normalization(hiddenState, decoderNormFinalWeights, decoderNormFinalBiases, settings.getEpsilon());
    }

    private int selectNextToken(float[] output)
    {
        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        float[] logits = Util.multiplyVectorByTransposedMatrix(output, tokenEmbeddings);

        // Find the index of the highest logit
        int tokenId = findBest(logits);

        // Print the generated token - It isn't perfect, because some words or letters represented by multiple tokens
        OUT.print(tokenizer.decode(Collections.singletonList(tokenId)));

        return tokenId;
    }

    /**
     * Clear stored values in all decoders to start a new session
     */
    public void clear()
    {
        for (TransformerDecoder decoder : decoders)
        {
            decoder.clear();
        }
    }
}
