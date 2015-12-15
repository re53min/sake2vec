package org.deeplearning4j.nnpractice;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

import static org.deeplearning4j.nnpractice.utils.*;

/**
 * Logistic Regression
 * Created by b1012059 on 2015/11/23.
 */
public class LogisticRegression {
    private static Logger log = LoggerFactory.getLogger(LogisticRegression.class);
    public int nIn;
    public int nOut;
    public double wIO[][];
    public double bias[];
    public int N;
    public Random rng;

    /**
     * LogisticRegression Constructor
     * @param nIn
     * @param nOut
     * @param N
     */
    public LogisticRegression(int nIn, int nOut, int N, Random rng){
        this.nIn = nIn;
        this.nOut = nOut;
        this.N = N;
        wIO = new double[nOut][nIn];
        bias = new double[nOut];

        //ランダムの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        //重み行列の初期化
        for(int i = 0; i < nOut; i++) {
            for(int j = 0; j < nIn; j++){
                wIO[i][j] = uniform(nIn, nOut, rng, null);
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
     * @param learningLate
     * @return
     */
    public double[] train(double input[], int teach[], double learningLate){
        double output[] = new double[nOut];
        double dOutput[] = new double[nOut];

        /*
        ロジスティック回帰の順方向計算
         */
        for(int i = 0; i < nOut; i++){
            output[i] = 0;
            for(int j = 0; j < nIn; j++){
                //入力とそれに対する重み行列の積
                output[i] += input[i] * wIO[i][j];
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
                wIO[i][j] += learningLate * dOutput[i] * input[j] / N;
            }
            //バイアスの更新
            bias[i] += learningLate * dOutput[i] / N;
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
        int nInput = 5;
        int nOutput = 2;
        int nTest = 2;
        int epochs = 500;
        double learningLate = 0.1;
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
                //{0., 1., 0., 0., 0., 0.},
                //{0., 0., 0., 0., 1., 0.}
                {0., 0., 1., 0., 0., 0.},
                {0., 0., 0., 0., 1., 0.}
        };

        double testOutput[][] = new double[nTest][nOutput];

        LogisticRegression logReg = new LogisticRegression(nInput, nOutput, inputData.length, rng);

        for(int epoch = 0; epoch < epochs; epoch++){
            for(int i = 0; i < inputData.length; i++){
                logReg.train(inputData[i], teachData[i], learningLate);
                //if(learningLate > 1e-5) learningLate *= 0.995;
                //log.info(String.valueOf(learningLate));
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