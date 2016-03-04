package org.deeplearning4j.nnpractice;

import java.util.Random;

import static org.deeplearning4j.nnpractice.utils.funSoftmax;
import static org.deeplearning4j.nnpractice.utils.uniform;

/**
 * Created by matsudate on 2016/03/04.
 */
public class SimpleLogisticRegression extends LogisticRegression{
    //private static Logger log = LoggerFactory.getLogger(LogisticRegression.class);
    private int nIn;
    private int nOut;
    public double wIO[][];
    private double bias[];
    private int N;
    private Random rng;

    /**
     * LogisticRegression Constructor
     * @param nIn
     * @param nOut
     * @param N
     */
    public SimpleLogisticRegression(int nIn, int nOut, int N, Random rng, String activation){
        super(nIn, nOut, N, rng, activation);
        this.nIn = nIn;
        this.nOut = nOut;
        this.N = N;
        this.wIO = new double[nOut][nIn];
        this.bias = new double[nOut];

        // log.info("Initialize LogisticLayer");

        //ランダムの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        //重み行列の初期化
        for(int i = 0; i < nOut; i++) {
            for(int j = 0; j < nIn; j++){
                wIO[i][j] = uniform(nIn, nOut, rng, activation);
            }
        }

        //バイアスの初期化
        for(int i = 0; i < nOut; i++){
            bias[i] = 0;
        }
    }

    /**
     * Training Method
     * @param input
     * @param teach
     * @param learningRate
     */
    public double[] train(double input[], int teach[], double learningRate){
        double output[] = new double[nOut];
        double dOutput[] = new double[nOut];

        /*
        ロジスティック回帰の順方向計算
         */
        for(int i = 0; i < nOut; i++){
            output[i] = 0;
            for(int j = 0; j < nIn; j++){
                //入力とそれに対する重み行列の積
                output[i] += wIO[i][j] * input[j];
            }
            //バイアス
            output[i] += bias[i];
        }
        //Softmax関数
        funSoftmax(output, nOut);

        /*
        ロジスティック回帰の逆方向学習
        確率的勾配降下法を用いてパラメータ更新
         */
        for(int i = 0; i < nOut; i++){
            //教師信号との誤差を求める
            dOutput[i] = teach[i] - output[i];

            for(int j = 0; j < nIn; j++){
                //重み行列の更新
                wIO[i][j] += learningRate * dOutput[i] * input[j] / N;
            }
            //バイアスの更新
            bias[i] += learningRate * dOutput[i] / N;
        }
        return dOutput;
    }

    /**
     * Testing Data Method
     * @param input テストデータ
     * @param output 出力
     */
    public void reconstruct(double input[], double output[]){
        //学習されたパラメータを使用した順方向計算
        for(int i = 0; i < nOut; i++){
            //初期化
            output[i] = 0;
            for(int j = 0; j < nIn; j++){
                output[i] += input[j] * wIO[i][j];
            }
            output[i] += bias[i];
        }

        //Softmax関数
        funSoftmax(output, nOut);

    }

    /**
     * Testing Method
     */
    private static void testLogisticRegression(){
        int nInput = 6;
        int nOutput = 2;
        int nTest = 2;
        int epochs = 500;
        double learningRate = 0.1;
        Random rng = new Random(123);

        double inputData[][] = {
                {1., 1., 1., 0., 0., 0.},
                {1., 0., 1., 0., 0., 0.},
                {1., 1., 1., 0., 0., 0.},
                {0., 0., 1., 1., 1., 0.},
                {0., 0., 1., 1., 0., 0.},
                {0., 0., 1., 1., 1., 0.}
        };

        int teachData[][] = {
                {1, 0},
                {1, 0},
                {1, 0},
                {0, 1},
                {0, 1},
                {0, 1}
        };

        double testData[][]= {

                {0., 1., 0., 0., 0., 1.},
                //{0., 0., 1., 1., 1., 0.}
        };

        double testOutput[][] = new double[nTest][nOutput];

        LogisticRegression logReg = new LogisticRegression(nInput, nOutput, inputData.length, rng, null);

        for(int epoch = 0; epoch < epochs; epoch++){
            for(int i = 0; i < inputData.length; i++){
                logReg.train(inputData[i], teachData[i], learningRate);
                //if(learningRate > 1e-5) learningRate *= 0.995;
                //log.info(String.valueOf(learningRate));
            }
        }


        System.out.println("-----------------TEST-----------------");
        for(int i = 0; i < nTest; i++){
            logReg.reconstruct(testData[i], testOutput[i]);
            for(int j = 0; j < nOutput; j++){
                System.out.print(testOutput[i][j] + " ");
            }
            System.out.println();
        }

    }

    public static void main(String args[]){
        testLogisticRegression();
    }
}

