package org.deeplearning4j.nnpractice;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

/**
 * バックプロパゲーションの練習問題3
 * API化したHiddenLayer及びLogisticRegressionを用いたBackPropagationの実現
 * テストデータがどの数字に一番近いかの分類
 * Created by b1012059 on 2015/11/22.
 */
public class BackPropagation4 {
    private static Logger log = LoggerFactory.getLogger(BackPropagation4.class);
    private int layerSize;
    private int nInput;
    private int hiddenSize[];
    private int nOutput;
    private int N;
    private HiddenLayer hLayer[];
    private LogisticRegression logisticLayer;
    private Random rng;

    public BackPropagation4(int INPUT, int HIDDEN[], int OUTPUT, int layer_size , int N, Random rng, String activation){
        //int inputLayer;

        this.N = N;
        this.layerSize = layer_size;
        this.hiddenSize = HIDDEN;
        this.hLayer = new HiddenLayer[layer_size];
        this.nInput = INPUT;
        this.nOutput = OUTPUT;

        //randomの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        //Hidden layerの初期化
        /*for(int i = 0; i < layer_size; i++){
            if(i == 0) inputLayer = nInput;
            else inputLayer = hiddenSize[i-1];

            //Hidden layer
            this.hLayer[i] = new HiddenLayer(inputLayer, hiddenSize[i], null, null, N, rng, activation);
        }*/
        this.hLayer[0] = new HiddenLayer(nInput, hiddenSize[0], null, null, N, rng, activation);

        this.logisticLayer = new LogisticRegression(hiddenSize[0], nOutput, N, rng);
    }

    /**
     * Training Method
     * @param input 入力データ
     * @param teach 教師データ
     * @param learningLate 学習率
     */
    public void train(double input[][], int teach[][], double learningLate){
        double[] hiddenInput;
        //double[] prevHiddenInput = new double[0];
        double[] outLayerInput;
        //double[] prevOutLayerInput = new double[0];
        double[] dOutput;


        for(int n = 0; n < N; n++) {
            /*for(int i = 0; i < layerSize; i++){
                if(i == 0) {
                    hiddenInput = new double[nInput];
                    outLayerInput = new double[hiddenSize[i]];

                    for(int j = 0; j < nInput; j++) hiddenInput[j] = input[n][j];
                    prevHiddenInput = hiddenInput;
                    hLayer[i].forwardCal(hiddenInput, outLayerInput);
                    prevOutLayerInput = outLayerInput;
                } else {
                    hiddenInput = new double[hiddenSize[i-1]];
                    for(int j = 0; j < hiddenSize[i-1]; j++) hiddenInput[j] = outLayerInput[j];
                    outLayerInput = new double[hiddenSize[i]];
                    hLayer[i].forwardCal(hiddenInput, outLayerInput);
                }
            }*/
            hiddenInput = new double[nInput];
            outLayerInput = new double[hiddenSize[0]];

            for(int j = 0; j < nInput; j++) hiddenInput[j] = input[n][j];
            hLayer[0].forwardCal(hiddenInput, outLayerInput);
            dOutput = logisticLayer.train(outLayerInput, teach[n], learningLate);

            hLayer[0].backwardCal(hiddenInput, null, outLayerInput, dOutput, logisticLayer.wIO, learningLate);
            /*for(int i = layerSize; i > 0; i--){
                if(i == layerSize) hLayer[i-1].backwardCal(hiddenInput, null, outLayerInput, dOutput, logisticLayer.wIO, learningLate);
                else hLayer[i-1].backwardCal(prevHiddenInput, null, prevOutLayerInput, dOutput, hLayer[i].wIO, learningLate);
            }*/
        }
    }

    /**
     * Testing Data Method
     * @param input
     * @param output
     */
    public void reconstruct(double input[], double output[]){
        //double prevHiddenInput[] = new double[0];
        double outLayerInput[] = new double[hiddenSize[0]];


        /*for(int i = 0; i < hiddenSize.length; i++){
            if(i == 0) {
                prevHiddenInput = new double[hiddenSize[i]];
                hLayer[i].forwardCal(input, prevHiddenInput);
                outLayerInput = prevHiddenInput;
            } else {
                outLayerInput = new double[hiddenSize[i]];
                hLayer[i].forwardCal(prevHiddenInput, outLayerInput);
            }
        }*/
        hLayer[0].forwardCal(input, outLayerInput);
        logisticLayer.reconstruct(outLayerInput, output);
    }

    /**
     * Tester Method
     */
    private static void testBackPropagation() {

        double inputData[][] = {
                //0
                {0, 0, 0, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 0, 0},

                //1
                {0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 1, 0, 0, 0,
                        0, 0, 1, 1, 0, 0, 0,
                        0, 0, 0, 1, 0, 0, 0,
                        0, 0, 0, 1, 0, 0, 0,
                        0, 0, 0, 1, 0, 0, 0,
                        0, 0, 0, 1, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 0, 0},

                //2
                {0, 0, 0, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0, 1, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 1, 0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 0, 0},

                //3
                {0, 0, 0, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0, 1, 0,
                        0, 0, 0, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0, 1, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 0, 0},

                //4
                {0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 1, 0, 0, 0,
                        0, 0, 1, 1, 0, 0, 0,
                        0, 1, 0, 1, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 0, 0, 1, 0, 0, 0,
                        0, 0, 0, 1, 0, 0, 0,
                        0, 0, 0, 1, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0},

                //5
                {0, 0, 0, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 1, 0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0, 1, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 0, 0},

                //6
                {0, 0, 0, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 1, 0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 0, 0},

                //7
                {0, 0, 0, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 0, 0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0, 0,
                        0, 0, 1, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0},

                //8
                {0, 0, 0, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 0, 0},

                //9
                {0, 0, 0, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0, 0, 0}
        };

        //教師データ
        int teachData[][] = {
                {1, 0},                  //0
                {0, 1},                  //1
                {0, 1},                  //2
                {0, 1},                  //3
                {1, 0},                  //4
                {0, 1},                  //5
                {1, 0},                  //6
                {0, 1},                  //7
                {1, 0},                  //8
                {1, 0}                   //9
        };

        //応用問題
        double testData[][] = {
                //C
                {0, 0, 0, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 1, 0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 0, 0},

                //E
                {0, 0, 0, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 1, 0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 0, 0,
                        0, 1, 0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 0, 0},

                //X
                {0, 0, 0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 0, 1, 0, 1, 0, 0,
                        0, 0, 0, 1, 0, 0, 0,
                        0, 0, 1, 0, 1, 0, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0, 0, 0},

                //A
                {0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 1, 0, 0, 0,
                        0, 0, 1, 0, 1, 0, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 0, 0},

                //Q
                {0, 0, 0, 0, 0, 0, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 0, 0, 0, 1, 0,
                        0, 1, 0, 1, 0, 1, 0,
                        0, 1, 0, 0, 1, 1, 0,
                        0, 1, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 0, 1}
        };

        /*double[][] inputData = {
                {0., 0.},
                {0., 1.},
                {1., 0.},
                {1., 1.},
        };

        /*int[][] teachData = {
                {0, 1},
                {1, 0},
                {1, 0},
                {0, 1},
        };

        // test data
        double[][] testData = {
                {0., 0.},
                {0., 1.},
                {1., 0.},
                {1., 1.},
        };*/

        int nInput = 63;
        int nHidden[] = {10};
        int nOutput = 2;
        //int nInput = 2;
        //int nHidden[] = {2};
        //int nOutput = 2;
        int nLayer = nHidden.length;
        int epochs = 200;
        int nTest = testData.length;
        int N = inputData.length;
        double learningLate = 0.1;
        Random rng = new Random(123);

        //インスタンス生成
        BackPropagation4 bp = new BackPropagation4(nInput, nHidden, nOutput, nLayer, N, rng, "ReLU");

        //Training
        for (int epoch = 0; epoch < epochs; epoch++) {
            bp.train(inputData, teachData, learningLate);
            //if(learningLate > 1e-5) learningLate *= 0.995;
            //log.info(String.valueOf(learningLate));
        }

        double testOutput[][] = new double[nTest][nOutput];
        //String testStr[] = {"C", "E", "X", "A", "Q", "0", "5", "9"};


        System.out.println("-----------------TEST-----------------");
        //Input test data
        for(int i = 0; i < nTest; i++){
            //System.out.println("Input: " + testStr[i]);
            bp.reconstruct(testData[i], testOutput[i]);
            for(int j = 0; j < nOutput; j++){
                System.out.print(testOutput[i][j] + " ");
            }
            System.out.println();
        }

        System.out.println("----------------FINISH----------------");
    }

    public static void main(String args[]){
        testBackPropagation();
    }
}
