package ai.demo.translator;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.TreeSet;
import static java.lang.Math.exp;
import static java.lang.Math.sqrt;

public class Util
{
    /**
     * Vector to vector addition
     */
    public static float[] addVectors(float[] vector1, float[] vector2)
    {
        float[] ret = new float[vector1.length];

        for (int i = 0; i < vector1.length; i++)
        {
            ret[i] = vector1[i] + vector2[i];
        }

        return ret;
    }

    /**
     * Dot product calculation (multiplying vector by vector)
     */
    public static float dotProduct(float[] vector1, float[] vector2)
    {
        float sum = 0;

        for (int i = 0; i < vector1.length; i++)
        {
            sum = sum + vector1[i] * vector2[i];
        }

        return sum;
    }

    /**
     * Multiply vector by a scalar
     */
    public static float[] multiplyVectorByScalar(float[] vector, float scalar)
    {
        float[] ret = new float[vector.length];

        for (int i = 0; i < vector.length; i++)
        {
            ret[i] = vector[i] * scalar;
        }

        return ret;
    }

    /**
     * Multiply vector by matrix
     */
    public static float[] multiplyVectorByMatrix(float[] vector, float[][] matrix)
    {
        float[] ret = new float[matrix[0].length];

        for (int col = 0; col < matrix[0].length; col++)
        {
            float sum = 0;

            for (int i = 0; i < vector.length; i++)
            {
                sum = sum + vector[i] * matrix[i][col];
            }

            ret[col] = sum;
        }

        return ret;
    }

    /**
     * Multiply vector by transposed matrix
     */
    public static float[] multiplyVectorByTransposedMatrix(float[] vector, float[][] matrix)
    {
        float[] ret = new float[matrix.length];

        for (int row = 0; row < matrix.length; row++)
        {
            float sum = 0;

            for (int i = 0; i < vector.length; i++)
            {
                sum = sum + vector[i] * matrix[row][i];
            }

            ret[row] = sum;
        }

        return ret;
    }

    /**
     * Split a vector to a matrix
     */
    public static float[][] splitVector(float[] vector, int count)
    {
        int size = vector.length / count;
        float[][] ret = new float[count][size];

        int segment = 0;
        int col = 0;
        for (float value : vector)
        {
            ret[segment][col] = value;

            if (col == size - 1)
            {
                col = 0;
                segment++;
            }
            else col++;
        }

        return ret;
    }

    /**
     * Merge the rows of a matrix to a single vector
     */
    public static float[] flattenMatrix(float[][] matrix)
    {
        float[] ret = new float[matrix.length * matrix[0].length];

        int i = 0;

        for (float[] row : matrix)
        {
            for (float value : row)
            {
                ret[i] = value;
                i++;
            }
        }

        return ret;
    }

    /**
     * Calculate average (mean) value
     */
    public static float average(float[] vector)
    {
        double sum = 0;

        for (float value : vector)
        {
            sum = sum + value;
        }

        return (float) sum / vector.length;
    }

    /**
     * Calculate the average difference -
     */
    public static float averageDiff(float[] values, float average, float epsilon)
    {
        float[] squareDiff = new float[values.length];

        for (int i = 0; i < values.length; i++)
        {
            float diff = values[i] - average;
            squareDiff[i] = diff * diff;
        }

        float averageSquareDiff = average(squareDiff);

        return (float) sqrt(averageSquareDiff + epsilon);
    }

    /**
     * Standard normalization - (value - avg) * sqrt( (value - avg)^2 + epsilon)
     */
    public static float[] normalize(float[] vector, float epsilon)
    {
        float average = average(vector);
        float averageDiff = averageDiff(vector, average, epsilon);

        float[] norm = new float[vector.length];

        for (int i = 0; i < vector.length; i++)
        {
            norm[i] = (vector[i] - average) / averageDiff;
        }

        return norm;
    }
}
