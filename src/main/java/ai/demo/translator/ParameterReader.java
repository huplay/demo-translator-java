package ai.demo.translator;

import java.io.File;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class ParameterReader
{
    public static float[] readVectorFile(String path, String fileName, int size)
    {
        return readParameterFile(path + "/" + fileName, size);
    }

    public static float[] readVectorFile(String path, String fileName, int size, boolean isPresent)
    {
        return isPresent ? readVectorFile(path, fileName, size) : null;
    }

    public static float[][] readMatrixFile(String path, String fileName, int rows, int cols)
    {
        float[] numbers = readParameterFile(path + "/" + fileName, rows * cols);
        return Util.splitVector(numbers, rows);
    }

    private static float[] readParameterFile(String fileName, int size)
    {
        fileName = fileName + ".dat";
        File file = new File(fileName);

        if (file.exists())
        {
            if (file.length() != (long) size * 4)
            {
                throw new RuntimeException("The size of the file (" + fileName + ", " + file.length() + ") is incorrect. Expected: " + size * 4);
            }

            return readFile(file);
        }
        else
        {
            // Handling files split into parts
            List<File> partFiles = findPartFiles(fileName);

            if ( ! partFiles.isEmpty())
            {
                float[][] parts = new float[partFiles.size()][size];

                // Read all the part files
                int i = 0;
                int sumSize = 0;
                for (File partFile : partFiles)
                {
                    parts[i] = readFile(partFile);
                    sumSize += parts[i].length;
                    i++;
                }

                if (sumSize != size)
                {
                    throw new RuntimeException("The sum size of the file parts (" + sumSize * 4 + ") is incorrect. Expected: " + (size * 4));
                }

                // Concatenate the parts into a single array
                float[] ret = new float[sumSize];

                int index = 0;
                for (float[] part : parts)
                {
                    for (float value : part)
                    {
                        ret[index] = value;
                        index++;
                    }
                }

                return ret;
            }
            else
            {
                throw new RuntimeException("Parameter file not found: " + fileName);
            }
        }
    }

    private static List<File> findPartFiles(String fileName)
    {
        List<File> partFiles = new ArrayList<>();

        int i = 1;
        while (true)
        {
            File partFile = new File(fileName + ".part" + i);

            if (partFile.exists()) partFiles.add(partFile);
            else break;

            i++;
        }

        return partFiles;
    }

    private static float[] readFile(File file)
    {
        int size = (int) (file.length() / 4);

        float[] array = new float[size];

        try (FileInputStream stream = new FileInputStream(file))
        {
            FileChannel inChannel = stream.getChannel();

            ByteBuffer buffer = inChannel.map(FileChannel.MapMode.READ_ONLY, 0, inChannel.size());

            buffer.order(ByteOrder.BIG_ENDIAN);
            FloatBuffer floatBuffer = buffer.asFloatBuffer();
            floatBuffer.get(array);
        }
        catch (Exception e)
        {
            throw new RuntimeException("Parameter file read error. (" + file.getName() + ")");
        }

        return array;
    }
}
