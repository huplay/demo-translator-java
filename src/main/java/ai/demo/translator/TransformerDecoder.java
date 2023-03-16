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

    private final float[][] selfQueryWeights;
    private final float[] selfQueryBiases;
    private final float[][] selfKeyWeights;
    private final float[] selfKeyBiases;
    private final float[][] selfValueWeights;
    private final float[] selfValueBiases;
    private final float[][] selfProjectionWeights;
    private final float[] selfProjectionBiases;
    private final float[] selfNormWeights;
    private final float[] selfNormBiases;

    private final float[][] crossQueryWeights;
    private final float[] crossQueryBiases;
    private final float[][] crossKeyWeights;
    private final float[] crossKeyBiases;
    private final float[][] crossValueWeights;
    private final float[] crossValueBiases;
    private final float[][] crossProjectionWeights;
    private final float[] crossProjectionBiases;
    private final float[] crossNormWeights;
    private final float[] crossNormBiases;

    private final float[][] mlpLayer1Weights;
    private final float[] mlpLayer1Biases;
    private final float[][] mlpLayer2Weights;
    private final float[] mlpLayer2Biases;
    private final float[] mlpNormWeights;
    private final float[] mlpNormBiases;

    private final List<float[][]> storedSelfKeys = new ArrayList<>();
    private final List<float[][]> storedSelfValues = new ArrayList<>();

    private final List<float[][]> storedCrossKeys = new ArrayList<>();
    private final List<float[][]> storedCrossValues = new ArrayList<>();

    /**
     * Initialization
     */
    public TransformerDecoder(int decoderId, Settings settings)
    {
        this.settings = settings;

        String path = settings.getPath() + "/decoders/decoder" + (decoderId + 1);
        int hiddenSize = settings.getHiddenSize();

        this.selfQueryWeights = readMatrixFile(path, "att.self.query.w", hiddenSize, hiddenSize);
        this.selfQueryBiases = readVectorFile(path, "att.self.query.b", hiddenSize, settings.hasAttentionQueryBias());
        this.selfKeyWeights = readMatrixFile(path, "att.self.key.w", hiddenSize, hiddenSize);
        this.selfKeyBiases = readVectorFile(path, "att.self.key.b", hiddenSize, settings.hasAttentionKeyBias());
        this.selfValueWeights = readMatrixFile(path, "att.self.value.w", hiddenSize, hiddenSize);
        this.selfValueBiases = readVectorFile(path, "att.self.value.b", hiddenSize, settings.hasAttentionValueBias());
        this.selfProjectionWeights = readMatrixFile(path, "att.self.proj.w", hiddenSize, hiddenSize);
        this.selfProjectionBiases = readVectorFile(path, "att.self.proj.b", hiddenSize, settings.hasAttentionProjectionBias());
        this.selfNormWeights = readVectorFile(path, "att.self.norm.w", hiddenSize);
        this.selfNormBiases = readVectorFile(path, "att.self.norm.b", hiddenSize);

        this.crossQueryWeights = readMatrixFile(path, "att.cross.query.w", hiddenSize, hiddenSize);
        this.crossQueryBiases = readVectorFile(path, "att.cross.query.b", hiddenSize, settings.hasAttentionQueryBias());
        this.crossKeyWeights = readMatrixFile(path, "att.cross.key.w", hiddenSize, hiddenSize);
        this.crossKeyBiases = readVectorFile(path, "att.cross.key.b", hiddenSize, settings.hasAttentionKeyBias());
        this.crossValueWeights = readMatrixFile(path, "att.cross.value.w", hiddenSize, hiddenSize);
        this.crossValueBiases = readVectorFile(path, "att.cross.value.b", hiddenSize, settings.hasAttentionValueBias());
        this.crossProjectionWeights = readMatrixFile(path, "att.cross.proj.w", hiddenSize, hiddenSize);
        this.crossProjectionBiases = readVectorFile(path, "att.cross.proj.b", hiddenSize, settings.hasAttentionProjectionBias());
        this.crossNormWeights = readVectorFile(path, "att.cross.norm.w", hiddenSize);
        this.crossNormBiases = readVectorFile(path, "att.cross.norm.b", hiddenSize);

        this.mlpLayer1Weights = readMatrixFile(path, "mlp.layer1.w", hiddenSize, hiddenSize * 4);
        this.mlpLayer1Biases = readVectorFile(path, "mlp.layer1.b", hiddenSize * 4, settings.hasMlpLayer1Bias());
        this.mlpLayer2Weights = readMatrixFile(path, "mlp.layer2.w", hiddenSize * 4, hiddenSize);
        this.mlpLayer2Biases = readVectorFile(path, "mlp.layer2.b", hiddenSize, settings.hasMlpLayer2Bias());
        this.mlpNormWeights = readVectorFile(path, "mlp.norm.w", hiddenSize);
        this.mlpNormBiases = readVectorFile(path, "mlp.norm.b", hiddenSize);
    }

    /**
     * Calculate keys ans values for all encoder outputs
     */
    public void calculateKeysAndValues(List<float[]> encoderOutputs)
    {
        for (float[] encoderOutput : encoderOutputs)
        {
            float[] key = applyWeight(encoderOutput, crossKeyWeights, crossKeyBiases);
            float[] value = applyWeight(encoderOutput, crossValueWeights, crossValueBiases);

            // Split the query, key and value vectors into pieces for all heads
            float[][] keys = Util.splitVector(key, settings.getDecoderHeadCount());
            float[][] values = Util.splitVector(value, settings.getDecoderHeadCount());

            // Store the keys and values (these will be available while the following tokens will be processed)
            storedCrossKeys.add(keys);
            storedCrossValues.add(values);
        }
    }

    /**
     * Decoder logic
     */
    public float[] execute(float[] hiddenState, List<float[]> encoderOutput)
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
        return normalization(hiddenState, selfNormWeights, selfNormBiases, settings.getEpsilon());
    }

    private float[] crossAttentionBlock(float[] inputHiddenState, List<float[]> encoderOutput)
    {
        // Attention layer
        float[] hiddenState = crossAttention(inputHiddenState, encoderOutput);

        // Add the original input state to the actual (residual connection)
        hiddenState = Util.addVectors(hiddenState, inputHiddenState);

        // Normalization
        return normalization(hiddenState, crossNormWeights, crossNormBiases, settings.getEpsilon());
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
        float[] query = applyWeight(hiddenState, selfQueryWeights, selfQueryBiases);
        float[] key = applyWeight(hiddenState, selfKeyWeights, selfKeyBiases);
        float[] value = applyWeight(hiddenState, selfValueWeights, selfValueBiases);

        // Split the query, key and value vectors into pieces for all heads
        float[][] queries = Util.splitVector(query, settings.getDecoderHeadCount());
        float[][] keys = Util.splitVector(key, settings.getDecoderHeadCount());
        float[][] values = Util.splitVector(value, settings.getDecoderHeadCount());

        // Store the keys and values (these will be available while the following tokens will be processed)
        storedSelfKeys.add(keys);
        storedSelfValues.add(values);

        float[][] sums = new float[settings.getDecoderHeadCount()][settings.getHiddenSize() / settings.getDecoderHeadCount()];

        // Scoring the previous tokens (including the actual), separately for all heads
        // Again: we have to score not only the previous, but the actual token as well
        // That is the reason of that we already added the actual key/value to the stored keys/values
        for (int head = 0; head < settings.getDecoderHeadCount(); head++)
        {
            // Calculate the scores
            float[] scores = new float[storedSelfKeys.size()];
            for (int pos = 0; pos < storedSelfKeys.size(); pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                scores[pos] = Util.dotProduct(queries[head], storedSelfKeys.get(pos)[head]) / settings.getDecoderScoreDividend();
            }

            // Softmax
            scores = softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedSelfKeys.size(); pos++)
            {
                float[] sum = Util.multiplyVectorByScalar(storedSelfValues.get(pos)[head], scores[pos]);
                sums[head] = Util.addVectors(sums[head], sum);
            }
        }

        // Concatenate the results for all heads
        float[] flatSums = Util.flattenMatrix(sums);

        // Apply the attention projection weights and biases
        return applyWeight(flatSums, selfProjectionWeights, selfProjectionBiases);
    }

    private float[] crossAttention(float[] hiddenState, List<float[]> encoderOutput)
    {
        // Calculate the query vector for the actual token:
        float[] query = applyWeight(hiddenState, crossQueryWeights, crossQueryBiases);

        // Split the query vector into pieces for all heads
        float[][] queries = Util.splitVector(query, settings.getDecoderHeadCount());

        float[][] sums = new float[settings.getDecoderHeadCount()][settings.getHiddenSize() / settings.getDecoderHeadCount()];

        // Scoring the encoder outputs, separately for all heads
        for (int head = 0; head < settings.getDecoderHeadCount(); head++)
        {
            // Calculate the scores
            float[] scores = new float[storedCrossKeys.size()];
            for (int pos = 0; pos < encoderOutput.size(); pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                scores[pos] = Util.dotProduct(queries[head], storedCrossKeys.get(pos)[head]) / settings.getDecoderScoreDividend();
            }

            // Softmax
            scores = softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < encoderOutput.size(); pos++)
            {
                float[] sum = Util.multiplyVectorByScalar(storedCrossValues.get(pos)[head], scores[pos]);
                sums[head] = Util.addVectors(sums[head], sum);
            }
        }

        // Concatenate the results for all heads
        float[] flatSums = Util.flattenMatrix(sums);

        // Apply the attention projection weights and biases
        return applyWeight(flatSums, crossProjectionWeights, crossProjectionBiases);
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
     * Clear the stored values after finishing the translation
     */
    public void clear()
    {
        storedSelfKeys.clear();
        storedSelfValues.clear();

        storedCrossKeys.clear();
        storedCrossValues.clear();
    }
}
