package org.deeplearning4j.nnpractice;

import java.util.Random;
import static org.deeplearning4j.nnpractice.utils.funSigmoid;

/**
 * AutoEncoderの実現。特に意味はない
 * Created by b1012059 on 2015/09/01.
 * @author Wataru Matsudate
 */
public class AutoEncoder {
    public int N;
    //encode用の各層配列(入力層、出力層)
    private double input[];
    public double output[];
    //ノイズ付加
    private double noiseX[];
    public Random rng;
    //出力層の重み配列
    public double wIO[][];
    //出力層の閾値配列
    public double threshOut[];
    //decode用の各層配列
    private double decodeIn[];
    private double decodeThO[];
    //学習率
    private final double alpha = 0.1;

    /**
     *
     * @param INPUT
     * @param OUTPUT
     */
    public AutoEncoder(int N, int INPUT, int OUTPUT, double wIO[][], double threshOut[], Random rng){
        this.N = N;
        this.input = new double[INPUT];
        this.noiseX = new double[INPUT];
        this.output = new double[OUTPUT];
        this.decodeIn = new double[INPUT];
        this.decodeThO = new double[INPUT];

        //randomの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng  = rng;


        //入力層→出力層の重み配列をランダム(-0.5~0.5)
        if(wIO == null) {
            this.wIO = new double[OUTPUT][INPUT];
            for (int i = 0; i < OUTPUT; i++) {
                for (int j = 0; j < INPUT; j++) {
                    this.wIO[i][j] = rng.nextDouble() - 0.5;
                }
            }
        } else {
            this.wIO = wIO;
        }

        //入力層→出力層の閾値配列を0で初期化
        if(threshOut == null){
            this.threshOut = new double[OUTPUT];
            for(int i = 0; i < OUTPUT; i++){
                this.threshOut[i] = 0;
            }
        } else {
            this.threshOut = threshOut;
        }

        //出力層→入力層の閾値配列を0で初期化
        for(int i = 0; i < INPUT; i++) {
            decodeThO[i] = 0;
        }
    }

    /**
     *ノイズ処理
     * 平均0、標準偏差1のガウス分布
     * @param x　inputデータ
     */
    public void noiseInput(double x[]){
        int lengthIn = x.length;

        for(int i = 0; i < lengthIn; i++){
            noiseX[i] = x[i] + rng.nextGaussian();
        }
    }

    /**
     *AutoEncoderにおける順方向計算(encode)の実現
     * 計算式:h = f(Wx+b)
     * なおf(x)はシグモイド関数
     * Wは重み行列、bはバイアス値
     */
    public void encoderCal(){
        int i,j;
        //各層の長さ
        int lengthIn = input.length;
        int lengthOut = output.length;
        //計算用temp
        double tmpData;

        //入力inputから符号outputを得る(encode)
        for(i = 0; i < lengthOut; i++){
            tmpData = -threshOut[i];
            for(j = 0; j < lengthIn; j++){
                // += input[j] * wIO[i][j];
                tmpData += noiseX[j] * wIO[i][j];
            }
            output[i] = funSigmoid(tmpData);
        }
    }

    /**
     *AutoEncoderにおける逆方向計算(decode)の実現
     * 計算式:y = f'(W'h+b')
     * なおf'(x)はシグモイド関数、W'=W^T
     * W'は重み行列、b'はバイアス値
     */
    public void decoderCal(){
        int i,j;
        //各層の長さ
        int lengthDeIn = decodeIn.length;
        int lengthOut = output.length;
        //計算用temp
        double tmpData;

        //符号outputから入力inputを復号する(decode)
        for(i = 0; i < lengthDeIn; i++) {
            tmpData = -decodeThO[i];
            for (j = 0; j < lengthOut; j++) {
                tmpData += output[j] * wIO[j][i];
            }
            decodeIn[i] = funSigmoid(tmpData);
        }
    }

    /**
     *
     * @param x
     * @param y
     */
    public void reconstruct(double x[], double y[]){
        double[] h = new double[output.length];

        noiseX = x;
        output = h;
        decodeIn = y;

        encoderCal();
        decoderCal();
    }

    /**
     * trainメソッド
     * 確立的勾配降下法を用いてパラメータ更新
     * @param x 学習データ
     */
    public void train(double x[]){
        //各パラメータの長さ
        int lengthOut = output.length;
        int lengthIn = input.length;

        noiseInput(x);
        encoderCal();
        decoderCal();

        double tempThO[] = new double[lengthOut];
        double tempDeThO[] = new double[lengthIn];

        //閾値decodeThOの変更(decodeのbiasの変更)
        for(int i = 0; i < lengthIn; i++){
            tempDeThO[i] = x[i] - decodeIn[i];
            decodeThO[i] += alpha * tempDeThO[i] / N;
        }

        //閾値threshOutの変更(encodeのbiasの変更)
        for(int i = 0; i < lengthOut; i++){
            tempThO[i] = 0;
            for(int j = 0; j < lengthIn; j++){
                tempThO[i] += wIO[i][j] * tempDeThO[j];
            }
            tempThO[i] *= output[i] * (1 - output[i]);
            threshOut[i] += alpha * tempThO[i] / N;
        }

        //重みwIOの変更
        for(int i = 0; i < lengthOut; i++){
            for(int j  = 0; j < lengthIn; j++){
                wIO[i][j] += alpha * (tempThO[i] * noiseX[j] + tempDeThO[j] * output[i]) / N;
            }
        }
    }

    /**
     *
     */
    private static void test_autoEncoder(){
        Random r = new Random(123);

        //入力データ
        double inputData[][] = {
                /*{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0}*/
                //0
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //1
                {0,  0,  0,  0,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  1,  1,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //2
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //3
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  0,  0,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //4
                {0,  0,  0,  0,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  1,  1,  0,  0,  0,
                        0,  1,  0,  1,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //5
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //6
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //7
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  0,  0,  0,  1,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  1,  0,  0,  0,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //8
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //9
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  0,  0,  0,  0,  1,  0,
                        0,  0,  0,  0,  0,  0,  0}
        };

        //testデータ
        double testData[][] = {
                /*{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}
                {1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0}*/
                //C
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //E
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  0,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //X
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  0,  1,  0,  1,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  1,  0,  1,  0,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //A
                {0,  0,  0,  0,  0,  0,  0,
                        0,  0,  0,  1,  0,  0,  0,
                        0,  0,  1,  0,  1,  0,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  0,  0},

                //Q
                {0,  0,  0,  0,  0,  0,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  0,  0,  1,  0,
                        0,  1,  0,  1,  0,  1,  0,
                        0,  1,  0,  0,  1,  1,  0,
                        0,  1,  1,  1,  1,  1,  0,
                        0,  0,  0,  0,  0,  0,  1}
        };

        int nIn = 63;
        int nOut = 10;
        int inputSize = inputData.length;

        //インスタンスの生成
        AutoEncoder ae = new AutoEncoder(inputSize, nIn, nOut, null, null, r);

        //train
        for(int count = 0; count < 200; count++){
            for (int i = 0; i < inputData.length; i++) {
                ae.train(inputData[i]);
            }
        }

        //weight print
        for(int i = 0; i < nOut; i++){
            for(int j = 0; j < nIn; j++){
                System.out.printf("%.5f", ae.wIO[i][j] + " ");
            }
            System.out.println("");
        }

        //test
        double reconstructed_X[][] = new double[testData.length][nIn];
        System.out.println("test AutoEncoder");
        for(int i = 0; i < testData.length; i++){
            ae.reconstruct(testData[i], reconstructed_X[i]);
            for(int j = 0; j < nIn; j++){
                System.out.printf("%.5f ", reconstructed_X[i][j]);
            }
            System.out.println();
        }
    }

    /**
     *
     * @param args
     */
    public static void main(String[] args) {
        test_autoEncoder();
    }
}
