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

        this.tokenEmbeddings = readMatrixFile(path, "encoders/input/wte", settings.getTokenCount(), hiddenSize);
        this.encoderPositionEmbeddings = readMatrixFile(path, "encoders/input/wpe", settings.getContextSize() + 2, hiddenSize);
        this.encoderNormWeights = readVectorFile(path, "encoders/input/norm.w", hiddenSize);
        this.encoderNormBiases = readVectorFile(path, "encoders/input/norm.b", hiddenSize);

        this.decoderPositionEmbeddings = readMatrixFile(path, "decoders/input/wpe", settings.getContextSize() + 2, hiddenSize);
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
        // Adding the START-OF-TEXT token to the beginning and END-OF-TEXT to the end
        // The output for this token will be the output of the whole encoder stack
        // (The output of the other tokens will be dropped, these will be used only by the attention mechanism.)
        inputTokens.add(0, settings.getStartOfTextToken());
        inputTokens.add(settings.getEndOfTextToken());

        List<float[]> encoderOutputs = executeEncoderStack(inputTokens);

        for (TransformerDecoder decoder : decoders)
        {
            decoder.calculateKeysAndValues(encoderOutputs);
        }

        // Collector of the generated new tokens (translation)
        List<Integer> result = new ArrayList<>();

        int token = settings.getEndOfTextToken();

        for (int pos = 0; pos < settings.getContextSize(); pos++)
        {
            // Add the last input token or the previously generated new token as input
            float[] hiddenState = decoderStack(pos, token, encoderOutputs);

            // The output will be the next new token
            token = selectNextToken(hiddenState);
            result.add(token);

            // Exit if the END_OF_TEXT token was chosen or the context size is reached
            if (token == settings.getEndOfTextToken()) break;
        }

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
            hiddenState = Util.addVectors(hiddenState, encoderPositionEmbeddings[pos + 2]);

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

    private float[] decoderStack(int pos, int token, List<float[]> encoderOutput)
    {
        float[] hiddenState = tokenEmbeddings[token];

        // Position embedding
        hiddenState = Util.addVectors(hiddenState, decoderPositionEmbeddings[pos + 2]);

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
        int tokenId = findBest(logits);

        // Print the generated token - It isn't perfect, because some words or letters represented by multiple tokens
        OUT.print(tokenizer.decode(Collections.singletonList(tokenId)));

        return tokenId;
    }

    private void clear()
    {
        for (TransformerDecoder decoder : decoders)
        {
            decoder.clear();
        }
    }
}
