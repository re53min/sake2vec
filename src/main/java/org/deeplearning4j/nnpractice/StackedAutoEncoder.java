package org.deeplearning4j.nnpractice;

import java.util.ArrayList;
import java.util.Random;

import static org.deeplearning4j.nnpractice.utils.updateLR;


/**
 * StackedAutoEncoderの実現。特に意味はない
 * Created by b1012059 on 2015/09/03.
 * @author Wataru Matsudate
 */
public class StackedAutoEncoder {

    private int layerSize;
    private int nIn;
    private int hiddenSize[];
    private int N;
    private HiddenLayer hLayer[];
    private AutoEncoder aeLayer[];
    private LogisticRegression logLayer;
    private Random rng;
    private String activation;

    public StackedAutoEncoder(int INPUT, int HIDDEN[], int OUTPUT, int N, Random rng, String activation){

        int inputLayer;
        this.N = N;
        this.nIn = INPUT;
        this.layerSize = HIDDEN.length;
        this.hiddenSize = HIDDEN;
        this.aeLayer = new AutoEncoder[layerSize];
        this.hLayer = new HiddenLayer[layerSize];
        this.activation = activation;

        //randomの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        //Hidden layerとAutoEncoder layerの初期化
        for(int i = 0; i < layerSize; i++){
            if(i == 0){
                inputLayer = INPUT;
            } else {
                inputLayer = this.hiddenSize[i-1];
            }

            //AutoEncoder layer
            this.aeLayer[i] = new AutoEncoder(this.N, inputLayer, this.hiddenSize[i],
                    null, null, rng, activation);
                    //hLayer[i].wIO, hLayer[i].bias, rng, activation);
        }
        this.logLayer = new LogisticRegression(this.hiddenSize[this.layerSize-1], OUTPUT, this.N, rng, activation);

    }

    /**
     * pre-trainingメソッド
     * AutoEncoderの学習を行う
     * @param inputData
     * @param learningRate
     * @param epochs
     * @param corruptionLevel
     */
    public void preTraining(int inputData[][], double learningRate, int epochs, double corruptionLevel){
        double[] inputLayer = new double[0];
        int prevInputSize;
        double[] prevInput;

        for(int i = 0; i < layerSize; i++){
            for(int epoch = 0; epoch < epochs; epoch++){
                for(int n = 0; n < N; n++){
                    for(int j = 0; j <= i; j++){
                        if(j == 0){
                            inputLayer = new double[nIn];
                            for(int k = 0; k < nIn; k++) inputLayer[k] = inputData[n][k];
                        } else {
                            if(j == 1) prevInputSize = nIn;
                            else prevInputSize = hiddenSize[j-2];

                            prevInput = new double[prevInputSize];
                            for(int k = 0; k < prevInputSize; k++) prevInput[k] = inputLayer[k];

                            inputLayer = new double[hiddenSize[j-1]];
                            //hLayer[j-1].sampleHgive(prevInput, inputLayer);
                            aeLayer[j-1].encoder(prevInput, inputLayer);
                        }
                    }
                    aeLayer[i].train(inputLayer, learningRate, corruptionLevel);
                }
            }
        }
    }

    /**
     * fine-tuningメソッド
     * pre-trainingの結果を使い、出力層を追加して
     * バックプロパゲーション
     * @param inputData
     * @param teach
     * @param learningRate
     * @param epochs
     * @param decayRate
     */
    public void fineTuning(int inputData[][], int teach[][], double learningRate, int epochs, double decayRate){
        int nLayer;
        double layerInput[] = new double[0];
        double prevLayerInput[];
        double dOutput[];
        double dhOutput[] = null;
        double defaultLR = learningRate;
        ArrayList<double[]> input = new ArrayList<>();
        ArrayList<double[]> output = new ArrayList<>();

        for (int epoch = 0; epoch < epochs; epoch++){
            for(int n = 0; n < N; n++){
                for(int i = 0; i < layerSize; i++){
                    if(i == 0){
                        prevLayerInput = new double[nIn];
                        for(int j = 0; j < nIn; j++) prevLayerInput[j] = inputData[n][j];
                        nLayer = prevLayerInput.length;
                    } else {
                        nLayer = hiddenSize[i-1];
                        prevLayerInput = new double[hiddenSize[i-1]];
                        for(int j = 0; j < hiddenSize[i-1]; j++) prevLayerInput[j] = layerInput[j];
                    }
                    input.add(prevLayerInput);
                    //Hidden layer
                    hLayer[i] = new HiddenLayer(nLayer, hiddenSize[i],
                            aeLayer[i].getWightIO(), aeLayer[i].getEncodeBias(), N, rng, activation);
                    layerInput = new double[hiddenSize[i]];
                    hLayer[i].forwardCal(prevLayerInput, layerInput);
                    output.add(layerInput);
                }

                //Output Layer
                dOutput = logLayer.train(layerInput, teach[n], learningRate);
                for(int k = layerSize-1; k <= 0; k--){
                    if(k == layerSize-1){
                        hLayer[k].backwardCal(input.get(k), dhOutput, output.get(k), dOutput, logLayer.wIO, learningRate);
                    } else {
                        hLayer[k].backwardCal(input.get(k), dhOutput, output.get(k), dhOutput, hLayer[k+1].getwIO(), learningRate);
                    }
                }

            }
            //Update LearningRate
            if(learningRate > 1E-3)
                learningRate = updateLR(defaultLR, decayRate, epoch);
            //System.out.println(learningRate);
        }
    }

    /**
     * Testing Data Method
     * @param input
     * @param output
     */
    public void reconstruct(int input[], double output[]) {
        double layerInput[] = new double[0];
        double prevLayerInput[] = new double[nIn];

        for (int i = 0; i < nIn; i++) prevLayerInput[i] = input[i];

        //Hidden Layer
        for (int i = 0; i < layerSize; i++) {
            layerInput = new double[hiddenSize[i]];
            hLayer[i].forwardCal(prevLayerInput, layerInput);

            if(i < layerSize){
                prevLayerInput = new double[hiddenSize[i]];
                for (int j = 0; j < hiddenSize[i]; j++) prevLayerInput[j] = layerInput[j];
            }
        }

        //Output Layer
        logLayer.reconstruct(layerInput, output);
    }

    private static void test_StackedaAE(){

        //入力データ
        int inputData[][] = {
                {1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}
                /*{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0}*/
        };

        //教師データ
        int teachData[][] = {
                {1, 0},
                {1, 0},
                {1, 0},
                {0, 1},
                {0, 1},
                {0, 1}
        };

        //testデータ
        int testData[][] = {
                {1, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
                /*
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                 */
        };

        int nInput = 10;
        int nHidden[] = {8, 6, 4};
        int nOutput = 2;
        Random rng = new Random(123);
        int epoch = 1000;
        double corruptionLevel = 0.3;
        double alpha = 0.1;
        double decayRate = 1E-2;

        //インスタンスの生成
        StackedAutoEncoder sAE = new StackedAutoEncoder(nInput, nHidden, nOutput,
                inputData.length, rng, "ReLU");

        //pre-training
        sAE.preTraining(inputData, alpha, epoch, corruptionLevel);
        //fine-tuning
        sAE.fineTuning(inputData, teachData, alpha, epoch, decayRate);

        //test
        int nTest = 2;
        double testOut[][] = new double[nTest][nOutput];

        for(int i = 0; i < nTest; i++) {
            sAE.reconstruct(testData[i], testOut[i]);
            for(int j = 0; j < nOutput; j++) {
                System.out.print(testOut[i][j] + " ");
            }
            System.out.println();
        }
    }

    public static void main(String args[]){
        test_StackedaAE();
    }
}