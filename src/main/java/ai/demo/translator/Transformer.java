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
    private final float[] encoderNormWeights;
    private final float[] encoderNormBiases;

    private final float[][] decoderPositionEmbeddings;
    private final float[] decoderNormWeights;
    private final float[] decoderNormBiases;

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

        int embeddingSize = settings.getContextSize() + settings.getSpecialTokenOffset();

        this.tokenEmbeddings = readMatrixFile(path, "input/wte", settings.getTokenCount(), hiddenSize);

        this.encoderPositionEmbeddings = readMatrixFile(path, "encoders/input/wpe", embeddingSize, hiddenSize);
        this.encoderNormWeights = readVectorFile(path, "encoders/input/norm.w", hiddenSize);
        this.encoderNormBiases = readVectorFile(path, "encoders/input/norm.b", hiddenSize);

        this.decoderPositionEmbeddings = readMatrixFile(path, "decoders/input/wpe", embeddingSize, hiddenSize);
        this.decoderNormWeights = readVectorFile(path, "decoders/input/norm.w", hiddenSize);
        this.decoderNormBiases = readVectorFile(path, "decoders/input/norm.b", hiddenSize);

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
        // Wrap the input between a START-OF-TEXT and END-OF-TEXT token
        inputTokens.add(0, settings.getStartOfTextToken());
        inputTokens.add(settings.getEndOfTextToken());

        // Process all input tokens by the encoders, it will produce a hidden state for all tokens
        List<float[]> encoderOutputs = executeEncoderStack(inputTokens);

        // Calculate the key and value vectors of the encoder outputs for all decoders
        // (It will be used by the cross attention mechanism of the decoders)
        for (TransformerDecoder decoder : decoders)
        {
            decoder.calculateKeysAndValues(encoderOutputs);
        }

        // Collector of the generated new tokens (translation)
        List<Integer> result = new ArrayList<>();

        // Feed the decoder stack with a starting input token.
        // (I used the START-OF-TEXT token, but it gives the same result with almost every other tokens.)
        int token = settings.getEndOfTextToken();

        for (int pos = 0; pos < settings.getContextSize(); pos++)
        {
            // Feed the decoder stack with the previously generated token (or with the initial one)
            float[] hiddenState = executeDecoderStack(pos, token, encoderOutputs);

            // Determine the token based on the hidden state produced by the decoder stack
            token = selectNextToken(hiddenState);
            result.add(token);

            // Exit if the END_OF_TEXT token was chosen
            if (token == settings.getEndOfTextToken()) break;

            // Print the generated token - It isn't perfect, because some words or letters represented by multiple tokens
            if (pos > 0) OUT.print(tokenizer.decode(Collections.singletonList(token)));
        }

        // Delete the stored values of the decoders
        clear();

        return result;
    }

    private List<float[]> executeEncoderStack(List<Integer> inputTokens)
    {
        List<float[]> hiddenStates = new ArrayList<>(inputTokens.size());

        for (int pos = 0; pos < inputTokens.size(); pos++)
        {
            // Word token embedding
            float[] hiddenState = tokenEmbeddings[inputTokens.get(pos)];

            // Position embedding
            hiddenState = Util.addVectors(hiddenState, encoderPositionEmbeddings[pos + settings.getSpecialTokenOffset()]);

            // Initial normalization
            hiddenState = normalization(hiddenState, encoderNormWeights, encoderNormBiases, settings.getEpsilon());

            hiddenStates.add(hiddenState);
        }

        // Encoder stack
        for (TransformerEncoder encoder : encoders)
        {
            for (int i = 0; i < inputTokens.size(); i++)
            {
                encoder.calculateKeysAndValues(hiddenStates.get(i));
            }

            for (int i = 0; i < inputTokens.size(); i++)
            {
                hiddenStates.set(i, encoder.execute(hiddenStates.get(i), hiddenStates.get(i)));
            }

            encoder.clear();
        }

        return hiddenStates;
    }

    private float[] executeDecoderStack(int pos, int token, List<float[]> encoderOutput)
    {
        // Word token embedding
        float[] hiddenState = tokenEmbeddings[token];

        // Position embedding
        hiddenState = Util.addVectors(hiddenState, decoderPositionEmbeddings[pos + settings.getSpecialTokenOffset()]);

        // Initial normalization
        hiddenState = normalization(hiddenState, decoderNormWeights, decoderNormBiases, settings.getEpsilon());

        // Decoder stack
        for (TransformerDecoder decoder : decoders)
        {
            hiddenState = decoder.execute(hiddenState, encoderOutput);
        }

        return hiddenState;
    }

    private int selectNextToken(float[] output)
    {
        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        float[] logits = Util.multiplyVectorByTransposedMatrix(output, tokenEmbeddings);

        // Find the index of the highest logit
        return findBest(logits);
    }

    private void clear()
    {
        for (TransformerDecoder decoder : decoders)
        {
            decoder.clear();
        }
    }
}
