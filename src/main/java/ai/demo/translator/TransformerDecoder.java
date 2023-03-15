package ai.demo.translator;

import java.util.ArrayList;
import java.util.List;
import static ai.demo.translator.ParameterReader.*;
import static ai.demo.translator.TransformerUtil.*;

/**
 * Decoder implementation for a decoder-only transformer
 */
public class TransformerDecoder
{
    private final Settings settings;

    private final float[][] queryWeights;
    private final float[] queryBiases;
    private final float[][] keyWeights;
    private final float[] keyBiases;
    private final float[][] valueWeights;
    private final float[] valueBiases;
    private final float[][] projectionWeights;
    private final float[] projectionBiases;
    private final float[] attNormWeights;
    private final float[] attNormBiases;

    private final float[][] encoderQueryWeights;
    private final float[] encoderQueryBiases;
    private final float[][] encoderKeyWeights;
    private final float[] encoderKeyBiases;
    private final float[][] encoderValueWeights;
    private final float[] encoderValueBiases;
    private final float[][] encoderProjectionWeights;
    private final float[] encoderProjectionBiases;
    private final float[] encoderAttNormWeights;
    private final float[] encoderAttNormBiases;

    private final float[][] mlpLayer1Weights;
    private final float[] mlpLayer1Biases;
    private final float[][] mlpLayer2Weights;
    private final float[] mlpLayer2Biases;
    private final float[] mlpNormWeights;
    private final float[] mlpNormBiases;

    private final List<float[][]> storedKeys = new ArrayList<>();
    private final List<float[][]> storedValues = new ArrayList<>();

    private final List<float[][]> storedEncoderKeys = new ArrayList<>();
    private final List<float[][]> storedEncoderValues = new ArrayList<>();

    /**
     * Initialization
     */
    public TransformerDecoder(int decoderId, Settings settings)
    {
        this.settings = settings;

        String path = settings.getPath() + "/decoders/decoder" + (decoderId + 1);
        int hiddenSize = settings.getHiddenSize();

        this.queryWeights = readMatrixFile(path, "att.query.w", hiddenSize, hiddenSize);
        this.queryBiases = readVectorFile(path, "att.query.b", hiddenSize, settings.hasAttentionQueryBias());
        this.keyWeights = readMatrixFile(path, "att.key.w", hiddenSize, hiddenSize);
        this.keyBiases = readVectorFile(path, "att.key.b", hiddenSize, settings.hasAttentionKeyBias());
        this.valueWeights = readMatrixFile(path, "att.value.w", hiddenSize, hiddenSize);
        this.valueBiases = readVectorFile(path, "att.value.b", hiddenSize, settings.hasAttentionValueBias());
        this.projectionWeights = readMatrixFile(path, "att.proj.w", hiddenSize, hiddenSize);
        this.projectionBiases = readVectorFile(path, "att.proj.b", hiddenSize, settings.hasAttentionProjectionBias());
        this.attNormWeights = readVectorFile(path, "att.norm.w", hiddenSize);
        this.attNormBiases = readVectorFile(path, "att.norm.b", hiddenSize);

        this.encoderQueryWeights = readMatrixFile(path, "att.query.encoder.w", hiddenSize, hiddenSize);
        this.encoderQueryBiases = readVectorFile(path, "att.query.encoder.b", hiddenSize, settings.hasAttentionQueryBias());
        this.encoderKeyWeights = readMatrixFile(path, "att.key.encoder.w", hiddenSize, hiddenSize);
        this.encoderKeyBiases = readVectorFile(path, "att.key.encoder.b", hiddenSize, settings.hasAttentionKeyBias());
        this.encoderValueWeights = readMatrixFile(path, "att.value.encoder.w", hiddenSize, hiddenSize);
        this.encoderValueBiases = readVectorFile(path, "att.value.encoder.b", hiddenSize, settings.hasAttentionValueBias());
        this.encoderProjectionWeights = readMatrixFile(path, "att.proj.encoder.w", hiddenSize, hiddenSize);
        this.encoderProjectionBiases = readVectorFile(path, "att.proj.encoder.b", hiddenSize, settings.hasAttentionProjectionBias());
        this.encoderAttNormWeights = readVectorFile(path, "att.norm.encoder.w", hiddenSize);
        this.encoderAttNormBiases = readVectorFile(path, "att.norm.encoder.b", hiddenSize);

        this.mlpLayer1Weights = readMatrixFile(path, "mlp.layer1.w", hiddenSize, hiddenSize * 4);
        this.mlpLayer1Biases = readVectorFile(path, "mlp.layer1.b", hiddenSize * 4, settings.hasMlpLayer1Bias());
        this.mlpLayer2Weights = readMatrixFile(path, "mlp.layer2.w", hiddenSize * 4, hiddenSize);
        this.mlpLayer2Biases = readVectorFile(path, "mlp.layer2.b", hiddenSize, settings.hasMlpLayer2Bias());
        this.mlpNormWeights = readVectorFile(path, "mlp.norm.w", hiddenSize);
        this.mlpNormBiases = readVectorFile(path, "mlp.norm.b", hiddenSize);
    }

    /**
     * Decoder logic
     */
    public float[] execute(float[] hiddenState, float[] encoderOutput)
    {
        // Self attention block
        hiddenState = selfAttentionBlock(hiddenState);

        // Cross-attention block
        hiddenState = crossAttentionBlock(hiddenState, encoderOutput);

        // Neuron layers
        return neuronBlock(hiddenState);
    }

    private float[] selfAttentionBlock(float[] inputHiddenState)
    {
        // Attention layer
        float[] hiddenState = selfAttention(inputHiddenState);

        // Add the original input state to the actual (residual connection)
        hiddenState = Util.addVectors(hiddenState, inputHiddenState);

        // Normalization
        return normalization(hiddenState, attNormWeights, attNormBiases, settings.getEpsilon());
    }

    private float[] crossAttentionBlock(float[] inputHiddenState, float[] encoderOutput)
    {
        // Attention layer
        float[] hiddenState = crossAttention(inputHiddenState, encoderOutput);

        // Add the original input state to the actual (residual connection)
        hiddenState = Util.addVectors(hiddenState, inputHiddenState);

        // Normalization
        return normalization(hiddenState, encoderAttNormWeights, encoderAttNormBiases, settings.getEpsilon());
    }

    private float[] neuronBlock(float[] inputHiddenState)
    {
        // Neuron layers
        float[] hiddenState = neuronLayers(inputHiddenState);

        // Add the original input state to the actual (residual connection)
        hiddenState = Util.addVectors(hiddenState, inputHiddenState);

        // Normalization
        return normalization(hiddenState, mlpNormWeights, mlpNormBiases, settings.getEpsilon());
    }

    private float[] selfAttention(float[] hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token:
        float[] query = applyWeight(hiddenState, queryWeights, queryBiases);
        float[] key = applyWeight(hiddenState, keyWeights, keyBiases);
        float[] value = applyWeight(hiddenState, valueWeights, valueBiases);

        // Split the query, key and value vectors into pieces for all heads
        float[][] queries = Util.splitVector(query, settings.getDecoderHeadCount());
        float[][] keys = Util.splitVector(key, settings.getDecoderHeadCount());
        float[][] values = Util.splitVector(value, settings.getDecoderHeadCount());

        // Store the keys and values (these will be available while the following tokens will be processed)
        storedKeys.add(keys);
        storedValues.add(values);

        float[][] sums = new float[settings.getDecoderHeadCount()][settings.getHiddenSize() / settings.getDecoderHeadCount()];

        // Scoring the previous tokens (including the actual), separately for all heads
        // Again: we have to score not only the previous, but the actual token as well
        // That is the reason of that we already added the actual key/value to the stored keys/values
        for (int head = 0; head < settings.getDecoderHeadCount(); head++)
        {
            // Calculate the scores
            float[] scores = new float[storedKeys.size()];
            for (int pos = 0; pos < storedKeys.size(); pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                scores[pos] = Util.dotProduct(queries[head], storedKeys.get(pos)[head]) / settings.getDecoderScoreDividend();
            }

            // Softmax
            scores = softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedKeys.size(); pos++)
            {
                float[] sum = Util.multiplyVectorByScalar(storedValues.get(pos)[head], scores[pos]);
                sums[head] = Util.addVectors(sums[head], sum);
            }
        }

        // Concatenate the results for all heads
        float[] flatSums = Util.flattenMatrix(sums);

        // Apply the attention projection weights and biases
        return applyWeight(flatSums, projectionWeights, projectionBiases);
    }

    private float[] crossAttention(float[] hiddenState, float[] encoderOutput)
    {
        // Calculate the query, key and value vectors for the actual token:
        float[] query = applyWeight(hiddenState, encoderQueryWeights, encoderQueryBiases);
        float[] key = applyWeight(encoderOutput, encoderKeyWeights, encoderKeyBiases);
        float[] value = applyWeight(encoderOutput, encoderValueWeights, encoderValueBiases);

        // Split the query, key and value vectors into pieces for all heads
        float[][] queries = Util.splitVector(query, settings.getDecoderHeadCount());
        float[][] keys = Util.splitVector(key, settings.getDecoderHeadCount());
        float[][] values = Util.splitVector(value, settings.getDecoderHeadCount());

        // Store the keys and values (these will be available while the following tokens will be processed)
        storedEncoderKeys.add(keys);
        storedEncoderValues.add(values);

        float[][] sums = new float[settings.getDecoderHeadCount()][settings.getHiddenSize() / settings.getDecoderHeadCount()];

        // Scoring the previous tokens (including the actual), separately for all heads
        // Again: we have to score not only the previous, but the actual token as well
        // That is the reason of that we already added the actual key/value to the stored keys/values
        for (int head = 0; head < settings.getDecoderHeadCount(); head++)
        {
            // Calculate the scores
            float[] scores = new float[storedKeys.size()];
            for (int pos = 0; pos < storedEncoderKeys.size(); pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                scores[pos] = Util.dotProduct(queries[head], storedEncoderKeys.get(pos)[head]) / settings.getDecoderScoreDividend();
            }

            // Softmax
            scores = softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedEncoderKeys.size(); pos++)
            {
                float[] sum = Util.multiplyVectorByScalar(storedEncoderValues.get(pos)[head], scores[pos]);
                sums[head] = Util.addVectors(sums[head], sum);
            }
        }

        // Concatenate the results for all heads
        float[] flatSums = Util.flattenMatrix(sums);

        // Apply the attention projection weights and biases
        return applyWeight(flatSums, encoderProjectionWeights, encoderProjectionBiases);
    }

    private float[] neuronLayers(float[] hiddenState)
    {
        // Layer 1: <hiddenSize> * 4 neurons (using a gelu activation function)
        hiddenState = applyWeight(hiddenState, mlpLayer1Weights, mlpLayer1Biases);
        for (int neuron = 0; neuron < settings.getHiddenSize() * 4; neuron++)
        {
            hiddenState[neuron] = gelu(hiddenState[neuron]);
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        return applyWeight(hiddenState, mlpLayer2Weights, mlpLayer2Biases);
    }

    /**
     * Clear stored values to start a new session
     */
    public void clear()
    {
        storedKeys.clear();
        storedValues.clear();
    }
}
