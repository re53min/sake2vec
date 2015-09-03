package org.deeplearning4j.nnpractice;

/**
 * AutoEncoderの練習。特に意味はない
 * Created by b1012059 on 2015/09/01.
 * @author Wataru Matsudate
 */
public class AutoEncoder {

    //encode用の各層配列(入力層、出力層)
    private int input[];
    private double output[];
    //出力層の重み配列
    private double wIO[][];
    //出力層の閾値配列
    private double threshOut[];
    //出力層の誤差配列
    //private double errorOut[];

    //decode用の各層配列
    private double decodeIn[];
    private double decodeThO[];
    //private double decodeErO[];

    //学習率
    private final double alpha = 0.1;
    //シグモイド関数の傾き
    private final double beta = 1.0;

    /**
     *
     * @param INPUT
     * @param OUTPUT
     */
    public AutoEncoder(int INPUT, int OUTPUT){
        int i, j;
        input = new int[INPUT];
        output = new double[OUTPUT];
        wIO = new double[OUTPUT][INPUT];
        threshOut = new double[OUTPUT];
        //errorOut = new double[OUTPUT];
        decodeIn = new double[INPUT];
        decodeThO = new double[INPUT];
        //decodeErO = new double[OUTPUT];


        //入力層→出力層の重み配列をランダム(-0.5~0.5)、閾値配列を0で初期化
        for(i = 0; i < OUTPUT; i++) {
            threshOut[i] = 0;
            for (j = 0; j < INPUT; j++) {
                wIO[i][j] = Math.random() - 0.5;
            }
        }

        //出力層→入力層の閾値配列を0で初期化
        for(i = 0; i < INPUT; i++) {
            decodeThO[i] = 0;
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
                tmpData += input[j] * wIO[i][j];
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
     *シグモイド関数
     * 入力された値を1~の間に補正する
     * @param tmpOutput
     * @return sigmoid 補正化された入力値
     */
    public double funSigmoid(double tmpOutput){
        double sigmoid = 1.0 / (1.0 + Math.exp(-beta * tmpOutput));
        return sigmoid;
    }

    /**
     *
     * @param y
     */
    public void reconstruct(double y[]){
        double[] h = new double[output.length];

        output = h;
        decodeIn = y;

        encoderCal();
        decoderCal();
    }

    /**
     *
     * @param N
     */
    public void train(int N){
        //各パラメータの長さ
        int lengthOut = output.length;
        int lengthIn = input.length;

        encoderCal();
        decoderCal();

        double tempThO[] = new double[lengthOut];
        double tempDeThO[] = new double[lengthIn];

        //閾値decodeThOの変更(decodeのbiasの変更)
        for(int i = 0; i < lengthIn; i++){
            tempDeThO[i] = input[i] - decodeIn[i];
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
                wIO[i][j] += alpha * (tempThO[i] * input[j] + tempDeThO[j] * output[i]) / N;
            }
        }
    }

    /**
     *
     */
    private static void test_autoEncoder(){

        //入力データ
        int inputData[][] = {
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0}
        };

        //testデータ
        int testData[][] = {
                {1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0}
        };

        //インスタンスの生成
        AutoEncoder ae = new AutoEncoder(20, 5);

        //train
        for(int count = 0; count < 100; count++){
            for (int i = 0; i < inputData.length; i++) {
                ae.input = inputData[i];
                ae.train(inputData.length);
            }
        }

        //test
        double reconstructed_X[][] = new double[testData.length][20];
        System.out.println("test autoencoder");
        for(int i = 0; i < testData.length; i++){
            ae.input = testData[i];
            ae.reconstruct(reconstructed_X[i]);
            for(int j = 0; j < 20; j++){
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
