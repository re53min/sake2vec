package org.deeplearning4j.nnpractice;

import java.util.ArrayList;
import java.util.Random;

import static org.deeplearning4j.nnpractice.utils.*;

/**
 * Created by b1012059 on 2015/09/04.
 */
public class DeepLearningTest {

    //各層の配列(入力層、中間層1、中間層2、出力層)
    private double input[];
    private double hidden1[];
    private double hidden2[];
    private double output[];
    //各層の重み配列
    private double wIH[][];
    private double wHH[][];
    private double wHO[][];
    //中間層1、中間層2、出力層の閾値配列
    private double threshHid1[];
    private double threshHid2[];
    private double threshOut[];
    //中間層1、中間層2、出力層の誤差配列
    private double errorHid1[];
    private double errorHid2[];
    private double errorOut[];
    //学習率
    private final double alpha = 1.0;
    //シグモイド関数の傾き
    private final double beta = 1.0;
    //layer size
    private int layerSize;
    private AutoEncoder aeLayer[];
    private int N;
    private int lengthIn;
    private int lengthHid[];
    private int lengthOut;

    private Random rng;



    public DeepLearningTest(int INPUT, int HIDDEN[], int OUTPUT, int layerSize, int N, Random rng){
        int inputLayer, i, j;
        this.N = N;
        this.lengthIn = INPUT;
        this.lengthOut = OUTPUT;
        this.layerSize = layerSize;
        this.lengthHid = HIDDEN;
        input = new double[INPUT];
        hidden1 = new double[HIDDEN[0]];
        hidden2 = new double[HIDDEN[1]];
        output = new double[OUTPUT];
        wIH = new double[HIDDEN[0]][INPUT];
        wHH = new double[HIDDEN[1]][HIDDEN[0]];
        wHO = new double[OUTPUT][HIDDEN[1]];
        threshHid1 = new double[HIDDEN[0]];
        threshHid2 = new double[HIDDEN[1]];
        threshOut = new double[OUTPUT];
        errorHid1 = new double[HIDDEN[0]];
        errorHid2 = new double[HIDDEN[1]];
        errorOut = new double[OUTPUT];
        this.aeLayer = new AutoEncoder[layerSize];

        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        for(i = 0; i < this.layerSize; i++) {
            if (i == 0) {
                inputLayer = input.length;
            } else {
                inputLayer = this.lengthHid[i-1];
            }
            this.aeLayer[i] = new AutoEncoder(this.N, inputLayer, this.lengthHid[i], null, null, rng);
        }

        //入力層→中間層1と中可能1→中間層2、中間層2→出力層の重み配列と閾値配列をランダム(-0.5~0.5)で初期化
        for(i = 0; i < this.lengthHid[0]; i++){
            threshHid1[i] = Math.random() - 0.5;
            for(j = 0; j < INPUT; j++){
                wIH[i][j] = Math.random() - 0.5;
            }
        }
        for(i = 0; i < this.lengthHid[1]; i++){
            threshHid2[i] = Math.random() - 0.5;
            for(j = 0; j < this.lengthHid[0]; j++){
                wHH[i][j] = Math.random() - 0.5;
            }
        }
        for(i = 0; i < this.lengthOut; i++) {
            threshOut[i] = Math.random() - 0.5;
            for (j = 0; j < this.lengthHid[1]; j++) {
                wHO[i][j] = Math.random() - 0.5;
            }
        }

    }

    public void pretrain(double x[][]){
        double layerInput[] = new double[0];
        double tempInput[];
        int lengthtempIn;

        //layerサイズ
        for(int i = 0; i < layerSize; i++) {
            //学習回数
            for (int count = 0; count < 1000; count++) {
                //input x1,x2,...,xN
                for (int n = 0; n < N; n++) {
                    //input layer
                    for (int j = 0; j <= i; j++) {
                        if (j == 0) {
                            layerInput = new double[lengthIn];
                            for(int k = 0; k < lengthIn; k++) layerInput[k] = x[n][k];
                        } else {
                            if(j == 1) lengthtempIn = lengthIn;
                            else lengthtempIn = lengthHid[i-1];

                            tempInput = new double[lengthtempIn];
                            for(int k = 0; k < lengthtempIn; k++) tempInput[k] = layerInput[k];

                            layerInput = new double[lengthHid[i-1]];
                            //layerInput = aeLayer[i-1].output;

                        }
                    }
                    aeLayer[i].train(layerInput);
                }
            }
        }
    }



    /**
     * BP前向き計算
     * 各層の出力はy = f(Σx*w - θ)で与えらえる
     * yは出力、xは入力、wは結合荷重、θは閾値、fはシグモイド関数
     */
    public void frontCal(){
        int i,j;
        //計算用temp
        double tmpData;

        //入力層→中間層1の計算
        for(i = 0; i < lengthHid[0]; i++){
            tmpData = -threshHid1[i];
            for(j = 0; j < lengthIn; j++){
                tmpData = tmpData + input[j] * wIH[i][j];
            }
            hidden1[i] = funSigmoid(tmpData);
        }
        //中間層1→中間層2の計算
        for(i = 0; i < lengthHid[1]; i++) {
            tmpData = -threshHid2[i];
            for (j = 0; j < lengthHid[0]; j++) {
                tmpData = tmpData + hidden1[j] * wHH[i][j];
            }
            hidden2[i] = funSigmoid(tmpData);
        }
        //中間層2→出力層の計算
        for(i = 0; i < lengthOut; i++){
            tmpData = -threshOut[i];
            for(j = 0; j < lengthHid[1]; j++){
                tmpData = tmpData + hidden2[j] * wHO[i][j];
            }
            output[i] = funSigmoid(tmpData);
        }
    }

    /**
     * 教師信号
     * @param teachData 正解データ
     * @return 正解データと出力結果を比較
     */
    public double teach(double teachData) {
        double t;
        //入力した値が閉じていれば「1 - 出力層」、閉じていなければ「0 - 出力層」
        t = teachData - output[0];

        return t;
    }

    /**
     * 中間層、出力層の誤差計算
     * 教師信号teachをもとに出力層の誤差を求める
     * その後出力層の誤差をもとに中間層の誤差を求める
     * @param teachData 正解データ
     */
    public void errorCal(double teachData[]){
        int i,j;
        //計算用temp
        double tmpData;

        //出力層の誤差計算
        for(i = 0; i < lengthOut; i++) {
            errorOut[i] = teach(teachData[i]) * dfunSigmoid(output[i]);
        }
        //中間層2の誤差計算
        for(i = 0; i < lengthHid[1]; i++){
            tmpData = 0.0;
            for(j = 0; j < lengthOut; j++){
                tmpData = tmpData + errorOut[j] * wHO[j][i];
            }
            errorHid2[i] = tmpData * dfunSigmoid(hidden2[i]);
        }
        //中間層1の誤差計算
        for(i = 0; i < lengthHid[0]; i++) {
            tmpData = 0.0;
            for (j = 0; j < lengthHid[1]; j++) {
                tmpData = tmpData + errorHid2[j] * wHH[j][i];
            }
            errorHid1[i] = tmpData * dfunSigmoid(hidden1[i]);
        }
    }

    /**
     * BP後ろ向き計算
     * 各層の誤差をもとに各層のパラメータ（結合荷重と閾値）を変更
     * パラメータの修正量は学習率alphaによって変化させる
     */
    public void backCal(){
        int i,j;

        //出力層と中間層2の学習
        for(i = 0; i < lengthOut; i++){
            threshOut[i] = threshOut[i] - alpha * errorOut[i];
            for(j = 0; j < lengthHid[1]; j++){
                wHO[i][j] = wHO[i][j] + alpha * errorOut[i] * hidden2[j];
            }
        }
        //中間層2と中間層1の学習
        for(i = 0; i < lengthHid[1]; i++){
            threshHid2[i] = threshHid2[i] - alpha * errorHid2[i];
            for(j = 0; j < lengthHid[0]; j++){
                wHH[i][j] = wHH[i][j] + alpha * errorHid2[i] * hidden1[j];
            }
        }
        //中間層1と入力層の学習
        for(i = 0; i < lengthHid[0]; i++){
            threshHid1[i] = threshHid1[i] - alpha * errorHid1[i];
            for(j = 0; j < lengthIn; j++){
                wIH[i][j] = wIH[i][j] + alpha * errorHid1[i] * input[j];
            }
        }
    }

    /**
     * 正解データと出力結果との二乗誤差を計算する
     * @param teach
     * @return 二乗誤差
     */
    public double calcError(double teach[]){
        double e = 0.0;
        int lengthOut = output.length;
        int i;

        for(i = 0; i < lengthOut; i++){
            e = e + Math.pow(teach(teach[i]), 2.0);
        }
        e = e * 0.5;
        return e;
    }

    /**
     * mainメソッド
     * @param args
     */
    public static void main(String[] args){

        System.out.println("デバッグ用１");

        int i, count = 0;
        int nInput = 63;
        int nHidden[] = {15, 5};
        int nOutput = 1;
        int nLayer = nHidden.length;
        ArrayList<Double> tmpError = new ArrayList<>();
        Random r = new Random(123);

        //入力データ
        double inputData[][] = {
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

        //教師データ
        double teachData[][] = {
                {1,0},                  //0
                {0,1},                  //1
                {0,1},                  //2
                {0,1},                  //3
                {1,0},                  //4
                {0,1},                  //5
                {1,0},                  //6
                {0,1},                  //7
                {1,0},                  //8
                {1,0}                   //9
        };

        //テストデータ
        double testData[][] = {
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
                        0,  0,  0,  0,  0,  0,  1},
        };

        //testデータ文字列
        String advanceResult[] = {"C", "E", "X", "A", "Q" };

        //DeepLearningTestのインスタンス生成
        DeepLearningTest bp = new DeepLearningTest(nInput, nHidden, nOutput, nLayer, inputData.length, r);

        //pretraining
        bp.pretrain(inputData);


        //BackPropagationによる学習
        while(true){

            //誤差
            double e = 0.0;

            for(i = 0; i < inputData.length; i++){

                bp.input = inputData[i];
                bp.frontCal();
                bp.errorCal(teachData[i]);
                bp.backCal();

                System.out.println("INPUT:" + i + " -> " + bp.output[0] + "(" + teachData[i][0] + ")");

                e = e + bp.calcError(teachData[i]);
            }

            count++;
            tmpError.add(e);
            System.out.println("Error = " + e);
            System.out.println(count + "回目");

            if(e < 0.001) {
                System.out.println("Error < 0.001");
                System.out.println("学習回数:" + count);
                break;
            }
        }

        //学習し終わったパーセプトロンにアルファベットを入力して判定させる
        System.out.println("アルファベットをパーセプトロンに入力します");
        for(i = 0; i < testData.length; i++) {
            bp.input = testData[i];
            bp.frontCal();
            System.out.println("INPUT:" + advanceResult[i] + " -> " + bp.output[0] + "(" + advanceResult[i] + ")");
        }
    }
}
