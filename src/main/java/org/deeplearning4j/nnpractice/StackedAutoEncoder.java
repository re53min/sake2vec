package org.deeplearning4j.nnpractice;

import java.util.Random;


/**
 * StackedAutoEncoderの実現。特に意味はない
 * Created by b1012059 on 2015/09/03.
 * @author Wataru Matsudate
 */
public class StackedAutoEncoder {

    private int layerSize;
    private int nIn;
    private int input[];
    private int hiddenSize[];
    private double output[];
    private int N;
    private SimpleHiddenLayer hLayer[];
    private AutoEncoder aeLayer[];
    private LogisticRegression logLayer;
    private Random rng;

    public StackedAutoEncoder(int INPUT, int HIDDEN[], int OUTPUT, int layerSize ,
                              int N, Random rng, String activation){

        int inputLayer;
        this.N = N;
        this.nIn = INPUT;
        this.layerSize = layerSize;
        this.hiddenSize = HIDDEN;
        this.aeLayer = new AutoEncoder[layerSize];
        this.hLayer = new SimpleHiddenLayer[layerSize];
        this.input = new int[INPUT];
        this.output = new double[OUTPUT];

        //randomの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        //Hidden layerとAutoEncoder layerの初期化
        for(int i = 0; i < layerSize; i++){
            if(i == 0){
                inputLayer = input.length;
            } else {
                inputLayer = this.hiddenSize[i-1];
            }

            //Hidden layer
            this.hLayer[i] = new SimpleHiddenLayer(inputLayer, this.hiddenSize[i], null, null, this.N, rng);

            //AutoEncoder layer
            this.aeLayer[i] = new AutoEncoder(this.N, inputLayer, this.hiddenSize[i],
                    hLayer[i].wIO, hLayer[i].bias, rng, activation);
        }
        this.logLayer = new LogisticRegression()

    }

    /**
     * pretrainingメソッド
     * AutoEncoder layerの学習を行う
     * @param inputData
     * @param learningRate
     * @param corruptionLevel
     */
    public void preTraining(int inputData[][], double learningRate, double corruptionLevel){
        double[] inputLayer = new double[0];
        int prevInputSize;
        double[] prevInput;

        for(int i = 0; i < layerSize; i++){
            for(int count = 0; count < 600; count++){
                for(int n = 0; n < N; n++){
                    for(int j = 0; j <= i; j++){
                        if(j == 0){
                            inputLayer = new double[input.length];
                            for(int k = 0; k < input.length; k++) inputLayer[k] = inputData[n][k];
                        } else {
                            if(j == 1) prevInputSize = input.length;
                            else prevInputSize = hiddenSize[j-2];

                            prevInput = new double[prevInputSize];
                            for(int k = 0; k < prevInputSize; k++) prevInput[k] = inputLayer[k];

                            inputLayer = new double[hiddenSize[j-1]];
                            hLayer[j-1].sampleHgive(prevInput, inputLayer);
                        }
                    }
                    aeLayer[i].train(inputLayer, learningRate, corruptionLevel);
                }
            }
        }
    }

    public void fineTuning(int inputData[][], int teach[], double learningRate, int epochs){
        double inputLayer[] = new double[0];
        double prevLayerInput[] = new double[0];

        for (int epoch = 0; epoch < epochs; epoch++){
            for(int n = 0; n < N; n++){
                for(int i = 0; i < layerSize; i++){
                    if(i == 0){
                        prevLayerInput = new double[nIn];
                        for(int j = 0; j < nIn; j++) prevLayerInput[j] = inputData[n][j];
                    } else {
                        prevLayerInput = new double[hiddenSize[i-1]];
                        for(int j=0; j<hiddenSize[i-1]; j++) prevLayerInput[j] = inputLayer[j];
                    }

                    inputLayer = new double[hiddenSize[i]];
                    hLayer[i].sampleHgive(prevLayerInput, inputLayer);
                }
            }
        }

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

        //testデータ
        int testData[][] = {
                {1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}
                /*
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                 */
        };

        int nInput = 10;
        int nHidden[] = {5, 2};
        int nOutput = 1;
        int nLayer = nHidden.length;
        Random rng = new Random(123);
        double corruptionLevel = 0.3;
        double alpha = 0.1;

        //インスタンスの生成
        StackedAutoEncoder sAE = new StackedAutoEncoder(nInput, nHidden, nOutput, nLayer,
                inputData.length, rng, null);

        //pretraining
        sAE.preTraining(inputData, alpha, corruptionLevel);


    }

    public static void main(String args[]){
        test_StackedaAE();
    }
}
