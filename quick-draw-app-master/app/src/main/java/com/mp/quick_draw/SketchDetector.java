package com.mp.quick_draw;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class SketchDetector {
    private final String TAG = this.getClass().getSimpleName();

    //tensorflowlite degiskeni
    private Interpreter tflite;

    private List<String> labelList;

    // Input byte buffer
    private ByteBuffer inputBuffer = null;

    // Output array [batch_size, 10]
    private float[][] mnistOutput = null;

    //assert dosyasındaki dosya adları
    private static final String MODEL_PATH = "model.tflite";
    private static final String LABEL_PATH = "class_names.txt";

    private static final int RESULTS_TO_SHOW = 1;

    //output size
    private static final int NUMBER_LENGTH = 100;

    //input size
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_IMG_SIZE_X = 28;
    private static final int DIM_IMG_SIZE_Y = 28;
    private static final int DIM_PIXEL_SIZE = 1;

    // byte sayisi
    private static final int BYTE_SIZE_OF_FLOAT = 4;

    private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });


    public SketchDetector(Activity activity) {
        try {
            tflite = new Interpreter(loadModelFile(activity));
            labelList = loadLabelList(activity);
            inputBuffer =
                    ByteBuffer.allocateDirect(
                            BYTE_SIZE_OF_FLOAT * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
            inputBuffer.order(ByteOrder.nativeOrder());
            mnistOutput = new float[DIM_BATCH_SIZE][NUMBER_LENGTH];
            Log.d(TAG, "Tensorflow Dosyasi Okundu");
        } catch (IOException e) {
            Log.e(TAG, "Tensorflow Dosyasi Okunamadi");
        }
    }

    //Tflite modeli olusturma
    protected void runInference() {
        tflite.run(inputBuffer, mnistOutput);
    }

    public String classify(Bitmap bitmap) {
        if (tflite == null) {
            Log.e(TAG, "Resim Siniflandirma Yapilamadi");
        }
        preprocess(bitmap);
        runInference();
        return printTopKLabels();
    }

    /**
     * Go through the output and find the number that was identified.
     *
     * @return the number that was identified (returns -1 if one wasn't found)
     */
    private int postprocess() {
        for (int i = 0; i < mnistOutput[0].length; i++) {
            float value = mnistOutput[0][i];
            Log.d(TAG, "Output for " + Integer.toString(i) + ": " + Float.toString(value));
            if (value == 1f) {
                return i;
            }
        }
        return -1;
    }

    private String printTopKLabels() {
        for (int i = 0; i < labelList.size(); ++i) {
            sortedLabels.add(
                    new AbstractMap.SimpleEntry<>(labelList.get(i), mnistOutput[0][i]));
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }
        String textToShow = "";
        final int size = sortedLabels.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            textToShow = label.getKey();
        }
        return textToShow;
    }

    //class_names.txt dosyasinden okuma
    private List<String> loadLabelList(Activity activity) throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(activity.getAssets().open(LABEL_PATH)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    //model.tflite dosyasini okuma
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Converts it into the Byte Buffer to feed into the model
     *
     * @param bitmap
     */
    private void preprocess(Bitmap bitmap) {
        if (bitmap == null || inputBuffer == null) {
            return;
        }
        inputBuffer.rewind();

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        // The bitmap shape should be 28 x 28
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int i = 0; i < pixels.length; ++i) {
            // Set 0 for white and 255 for black pixels
            int pixel = pixels[i];
            // The color of the input is black so the blue channel will be 0xFF.
            int channel = pixel & 0xff;
            inputBuffer.putFloat(0xff - channel);
        }
    }
}
