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
public class MultiLayerPerceptron {
    private static Logger log = LoggerFactory.getLogger(MultiLayerPerceptron.class);
    private int nInput;
    private int hiddenSize;
    private int nOutput;
    private int N;
    private HiddenLayer hLayer;
    private LogisticRegression logisticLayer;
    private Random rng;

    public MultiLayerPerceptron(int INPUT, int HIDDEN, int OUTPUT, int N, Random rng, String activation){

        this.N = N;
        this.hiddenSize = HIDDEN;
        this.nInput = INPUT;
        this.nOutput = OUTPUT;

        //randomの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        this.hLayer = new HiddenLayer(this.nInput, this.hiddenSize, null, null, N, rng, activation);
        this.logisticLayer = new LogisticRegression(this.hiddenSize, this.nOutput, N, rng,activation);
    }

    /**
     * Training Method
     * @param input 入力データ
     * @param teach 教師データ
     * @param learningLate 学習率
     */
    public void train(double input[][], int teach[][], double learningLate){
        double[] hiddenInput;
        double[] outLayerInput;
        double[] dOutput;


        for(int n = 0; n < N; n++) {
            hiddenInput = new double[nInput];
            outLayerInput = new double[hiddenSize];

            for(int j = 0; j < nInput; j++) hiddenInput[j] = input[n][j];
            hLayer.forwardCal(hiddenInput, outLayerInput);
            dOutput = logisticLayer.train(outLayerInput, teach[n], learningLate);

            hLayer.backwardCal(hiddenInput, null, outLayerInput, dOutput, logisticLayer.wIO, learningLate);
        }
    }

    /**
     * Testing Data Method
     * @param input
     * @param output
     */
    public void reconstruct(double input[], double output[]){
        double outLayerInput[] = new double[hiddenSize];

        hLayer.forwardCal(input, outLayerInput);
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
        int nHidden = 10;
        int nOutput = 2;
        //int nInput = 2;
        //int nHidden[] = {2};
        //int nOutput = 2;
        //int nLayer = 1;
        int epochs = 200;
        int nTest = testData.length;
        int N = inputData.length;
        double learningLate = 0.1;
        Random rng = new Random(123);

        //インスタンス生成
        MultiLayerPerceptron bp = new MultiLayerPerceptron(nInput, nHidden, nOutput, N, rng, "ReLU");

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
