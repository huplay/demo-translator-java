package ai.demo.translator;

import java.util.Comparator;
import java.util.List;
import java.util.TreeSet;

import static java.lang.Math.*;

public class TransformerUtil
{
    /**
     * Applying weights using vector by matrix multiplication plus adding biases
     */
    public static float[] applyWeight(float[] vector, float[][] weights, float[] biases)
    {
        float[] result = Util.multiplyVectorByMatrix(vector, weights);

        if (biases != null)
        {
            result = Util.addVectors(result, biases);
        }

        return result;
    }

    /**
     * Standard normalization with applying weights and biases
     */
    public static float[] normalization(float[] vector, float[] weights, float[] biases, float epsilon)
    {
        // Standard normalization
        float[] result = Util.normalize(vector, epsilon);

        // Applying the trained weights and biases
        for (int i = 0; i < vector.length; i++)
        {
            result[i] = result[i] * weights[i] + biases[i];
        }

        return result;
    }

    /**
     * Gaussian Error Linear Unit (GELU) cumulative distribution activation function (approximate implementation)
     * Original paper: https://paperswithcode.com/method/gelu
     */
    public static float gelu(float value)
    {
        return (float) (0.5 * value * (1 + tanh(sqrt(2 / PI) * (value + 0.044715 * value * value * value))));
    }

    /**
     * Calculate softmax - rescale the values into a range between 0 and 1
     */
    public static float[] softmax(float[] vector)
    {
        double total = 0;
        for (float value : vector)
        {
            total = total + exp(value);
        }

        float[] ret = new float[vector.length];

        for (int i = 0; i < vector.length; i++)
        {
            ret[i] = (float) (exp(vector[i]) / total);
        }

        return ret;
    }

    /**
     * Calculate softmax on IndexedValue list - rescale the values into a range between 0 and 1
     */
    public static float[] softmax(List<IndexedValue> values)
    {
        double total = 0;
        for (IndexedValue value : values)
        {
            total = total + exp(value.value);
        }

        float[] ret = new float[values.size()];

        for (int i = 0; i < values.size(); i++)
        {
            ret[i] = (float) (exp(values.get(i).value) / total);
        }

        return ret;
    }

    /**
     * Weighted random selection from list of probabilities
     */
    public static int weightedRandomPick(float[] probabilities)
    {
        float sum = 0;
        float[] cumulativeProbabilities = new float[probabilities.length];

        for (int i = 0; i < probabilities.length; i++)
        {
            sum = sum + probabilities[i] * 100;
            cumulativeProbabilities[i] = sum;
        }

        int random = (int)(Math.random() * sum);

        int index = 0;
        for (int i = 0; i < probabilities.length; i++)
        {
            if (random < cumulativeProbabilities[i]) break;

            index ++;
        }

        return index;
    }

    /**
     * Sort values to reversed order and filter out the lowest values (retain the top [count] values)
     */
    public static int findBest(float[] values)
    {
        TreeSet<IndexedValue> indexedValues = new TreeSet<>(new ReverseComparator());
        for (int i = 0; i < values.length; i++)
        {
            indexedValues.add(new IndexedValue(values[i], i));
        }

        return indexedValues.first().index;
    }

    /**
     * Holder of a value with the index (position of the element)
     */
    public static class IndexedValue
    {
        public final float value;
        public final int index;

        public IndexedValue(float value, int index)
        {
            this.value = value;
            this.index = index;
        }
    }

    /**
     * Comparator for IndexedValue to achieve reverse ordering
     */
    private static class ReverseComparator implements Comparator<IndexedValue>
    {
        public int compare(IndexedValue a, IndexedValue b)
        {
            return Float.compare(b.value, a.value);
        }
    }
}
