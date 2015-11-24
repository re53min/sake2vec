package org.deeplearning4j.nnpractice;

import java.util.Random;


/**
 * StackedAutoEncoderの実現。特に意味はない
 * Created by b1012059 on 2015/09/03.
 * @author Wataru Matsudate
 */
public class StackedAutoEncoder {

    private int layerSize;
    private int input[];
    private int hiddenSize[];
    private double output[];
    private int N;
    private HiddenLayerSimple hLayer[];
    private AutoEncoder aeLayer[];
    private Random rng;

    public StackedAutoEncoder(int INPUT, int HIDDEN[], int OUTPUT, int layer_size , int N, Random rng, String activation){
        int inputLayer;

        this.N = N;
        this.layerSize = layer_size;
        this.hiddenSize = HIDDEN;
        this.aeLayer = new AutoEncoder[layer_size];
        this.hLayer = new HiddenLayerSimple[layer_size];
        input = new int[INPUT];
        output = new double[OUTPUT];

        //randomの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        //Hidden layerとAutoEncoder layerの初期化
        for(int i = 0; i < layer_size; i++){
            if(i == 0){
                inputLayer = input.length;
            } else {
                inputLayer = this.hiddenSize[i-1];
            }

            //Hidden layer
            this.hLayer[i] = new HiddenLayerSimple(inputLayer, this.hiddenSize[i], null, null, this.N, rng);

            //AutoEncoder layer
            this.aeLayer[i] = new AutoEncoder(this.N, inputLayer, this.hiddenSize[i], hLayer[i].wIO, hLayer[i].bias, rng);
        }

    }

    /**
     * pretrainingメソッド
     * AutoEncoder layerの学習を行う
     * @param x 学習データ
     */
    public void pretrain(int x[][]){
        double[] layerInput = new double[0];
        int prevInputSize;
        double[] prevInput;

        for(int i = 0; i < layerSize; i++){
            for(int count = 0; count < 600; count++){
                for(int n = 0; n < N; n++){
                    for(int j = 0; j <= i; j++){
                        if(j == 0){
                            layerInput = new double[input.length];
                            for(int k = 0; k < input.length; k++) layerInput[k] = x[n][k];
                        } else {
                            if(j == 1) prevInputSize = input.length;
                            else prevInputSize = hiddenSize[j-2];

                            prevInput = new double[prevInputSize];
                            for(int k = 0; k < prevInputSize; k++) prevInput[k] = layerInput[k];

                            layerInput = new double[hiddenSize[j-1]];
                            hLayer[j-1].sampleHgive(prevInput, layerInput);
                        }
                    }
                    aeLayer[i].train(layerInput);
                }
            }
        }
    }

    public void finetune(int input[][], int teach[]){


    }

    private static void test_StackedaAE(){
        int nInput = 10;
        int nHidden[] = {5, 2};
        int nOutput = 1;
        int nLayer = nHidden.length;
        Random rng = new Random(123);

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

        //インスタンスの生成
        StackedAutoEncoder sAE = new StackedAutoEncoder(nInput, nHidden, nOutput, nLayer, inputData.length, rng, null);

        //pretraining
        sAE.pretrain(inputData);


    }

    public static void main(String args[]){
        test_StackedaAE();
    }
}
