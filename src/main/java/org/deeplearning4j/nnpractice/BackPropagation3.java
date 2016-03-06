package org.deeplearning4j.nnpractice;

import org.deeplearning4j.CreateGraph;

import javax.swing.*;
import java.util.ArrayList;

/**
 * バックプロパゲーションの練習問題3
 * 数字が閉じているか閉じていないかの判別
 * 隠れ層2の多層パーセプトロン
 * Created by b1012059 on 2015/05/27.
 * @author b1012059 Wataru Matsudate
 */

public class BackPropagation3 {

    //各層の配列(入力層、中間層1、中間層2、出力層)
    private int input[];
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


    /**
     * コンストラクタ
     * @param INPUT
     * @param HIDDEN1
     * @param HIDDEN2
     * @param OUTPUT
     */
    public BackPropagation3(int INPUT, int HIDDEN1, int HIDDEN2, int OUTPUT){
        int i, j;
        input = new int[INPUT];
        hidden1 = new double[HIDDEN1];
        hidden2 = new double[HIDDEN2];
        output = new double[OUTPUT];
        wIH = new double[HIDDEN1][INPUT];
        wHH = new double[HIDDEN2][HIDDEN1];
        wHO = new double[OUTPUT][HIDDEN2];
        threshHid1 = new double[HIDDEN1];
        threshHid2 = new double[HIDDEN2];
        threshOut = new double[OUTPUT];
        errorHid1 = new double[HIDDEN1];
        errorHid2 = new double[HIDDEN2];
        errorOut = new double[OUTPUT];

        //入力層→中間層1と中可能1→中間層2、中間層2→出力層の重み配列と閾値配列をランダム(-0.5~0.5)で初期化
        for(i = 0; i < HIDDEN1; i++){
            threshHid1[i] = Math.random() - 0.5;
            for(j = 0; j < INPUT; j++){
                wIH[i][j] = Math.random() - 0.5;
            }
        }
        for(i = 0; i < HIDDEN2; i++){
            threshHid2[i] = Math.random() - 0.5;
            for(j = 0; j < HIDDEN1; j++){
                wHH[i][j] = Math.random() - 0.5;
            }
        }
        for(i = 0; i < OUTPUT; i++){
            threshOut[i] = Math.random() - 0.5;
            for(j = 0; j < HIDDEN2; j++){
                wHO[i][j] = Math.random() - 0.5;
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
        //各層の長さ
        int lengthIn = input.length;
        int lengthHid1 = hidden1.length;
        int lengthHid2 = hidden2.length;
        int lengthOut = output.length;
        //計算用temp
        double tmpData;

        //入力層→中間層1の計算
        for(i = 0; i < lengthHid1; i++){
            tmpData = -threshHid1[i];
            for(j = 0; j < lengthIn; j++){
                tmpData = tmpData + input[j] * wIH[i][j];
            }
            hidden1[i] = funSigmoid(tmpData);
        }
        //中間層1→中間層2の計算
        for(i = 0; i < lengthHid2; i++) {
            tmpData = -threshHid2[i];
            for (j = 0; j < lengthHid1; j++) {
                tmpData = tmpData + hidden1[j] * wHH[i][j];
            }
            hidden2[i] = funSigmoid(tmpData);
        }
        //中間層2→出力層の計算
        for(i = 0; i < lengthOut; i++){
            tmpData = -threshOut[i];
            for(j = 0; j < lengthHid2; j++){
                tmpData = tmpData + hidden2[j] * wHO[i][j];
            }
            output[i] = funSigmoid(tmpData);
        }
    }

    /**
     * tmpDataを引数としたシグモイド関数の計算
     * @param tmpData 入力と結合荷重の総和から閾値を引いた値
     * @return sigmoid(x) = 1 / (1 + exp(-x))によって計算した数値
     */
    public double funSigmoid(double tmpData){
        double sigmoid = 1.0 / (1.0 + Math.exp(-beta * tmpData));
        return sigmoid;
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
        //各層の長さ
        int lengthHid1 = hidden1.length;
        int lengthHid2 = hidden2.length;
        int lengthOut = output.length;
        //計算用temp
        double tmpData;

        //出力層の誤差計算
        for(i = 0; i < lengthOut; i++) {
            errorOut[i] = teach(teachData[i]) * output[i] * (1.0 - output[i]);
        }
        //中間層2の誤差計算
        for(i = 0; i < lengthHid2; i++){
            tmpData = 0.0;
            for(j = 0; j < lengthOut; j++){
                tmpData = tmpData + errorOut[j] * wHO[j][i];
            }
            errorHid2[i] = hidden2[i] * tmpData * (1 - hidden2[i]);
        }
        //中間層1の誤差計算
        for(i = 0; i < lengthHid1; i++) {
            tmpData = 0.0;
            for (j = 0; j < lengthHid2; j++) {
                tmpData = tmpData + errorHid2[j] * wHH[j][i];
            }
            errorHid1[i] = hidden1[i] * tmpData * (1 - hidden1[i]);
        }
    }

    /**
     * BP後ろ向き計算
     * 各層の誤差をもとに各層のパラメータ（結合荷重と閾値）を変更
     * パラメータの修正量は学習率alphaによって変化させる
     */
    public void backCal(){
        int i,j;
        //各層の長さ
        int lengthIn = input.length;
        int lengthHid1 = hidden1.length;
        int lengthHid2 = hidden2.length;
        int lengthOut = output.length;

        //出力層と中間層2の学習
        for(i = 0; i < lengthOut; i++){
            threshOut[i] = threshOut[i] - alpha * errorOut[i];
            for(j = 0; j < lengthHid2; j++){
                wHO[i][j] = wHO[i][j] + alpha * errorOut[i] * hidden2[j];
            }
        }
        //中間層2と中間層1の学習
        for(i = 0; i < lengthHid2; i++){
            threshHid2[i] = threshHid2[i] - alpha * errorHid2[i];
            for(j = 0; j < lengthHid1; j++){
                wHH[i][j] = wHH[i][j] + alpha * errorHid2[i] * hidden1[j];
            }
        }
        //中間層1と入力層の学習
        for(i = 0; i < lengthHid1; i++){
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

        //グラフ作成用
        CreateGraph frame = new CreateGraph("中間層2層(4,4)における二乗誤差の推移", "学習回数(n)", "二乗誤差(e)");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setBounds(10, 10, 800, 500);
        frame.setTitle("BackPropagation3");
        //error用
        ArrayList<Double> tmpError = new ArrayList<>();
        ArrayList<Double> tmpZero = new ArrayList<>();
        ArrayList<Double> tmpOne = new ArrayList<>();
        ArrayList<Double> tmpTwo = new ArrayList<>();
        ArrayList<Double> tmpThree = new ArrayList<>();
        ArrayList<Double> tmpFour = new ArrayList<>();
        ArrayList<Double> tmpFive = new ArrayList<>();
        ArrayList<Double> tmpSix = new ArrayList<>();
        ArrayList<Double> tmpSeven = new ArrayList<>();
        ArrayList<Double> tmpEight = new ArrayList<>();
        ArrayList<Double> tmpNine = new ArrayList<>();


        //入力データ
        int inputData[][] = {
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
                {1.0},                  //0
                {0.0},                  //1
                {0.0},                  //2
                {0.0},                  //3
                {1.0},                  //4
                {0.0},                  //5
                {1.0},                  //6
                {0.0},                  //7
                {1.0},                  //8
                {1.0},                  //9
        };

        //応用問題
        int advanceData[][] = {
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

        //応用問題入力文字列
        String advanceResult[] = {"C", "E", "X", "A", "Q" };

        //BackPropagationのインスタンス生成
        BackPropagation3 bp = new BackPropagation3(63, 30, 10, 1);

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
                switch (i){
                    case 0:
                        tmpZero.add(bp.output[0]);
                    case 1:
                        tmpOne.add(bp.output[0]);
                    case 2:
                        tmpTwo.add(bp.output[0]);
                    case 3:
                        tmpThree.add(bp.output[0]);
                    case 4:
                        tmpFour.add(bp.output[0]);
                    case 5:
                        tmpFive.add(bp.output[0]);
                    case 6:
                        tmpSix.add(bp.output[0]);
                    case 7:
                        tmpSeven.add(bp.output[0]);
                    case 8:
                        tmpEight.add(bp.output[0]);
                    case 9:
                        tmpNine.add(bp.output[0]);
                    default:
                }

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
        for(i = 0; i < advanceData.length; i++) {
            bp.input = advanceData[i];
            bp.frontCal();
            System.out.println("INPUT:" + advanceResult[i] + " -> " + bp.output[0] + "(" + advanceResult[i] + ")");
        }
        //グラフ表示
        //frame.getContentPane().add(frame.createGraphPanel2(count, tmpError, tmpZero, tmpOne, tmpTwo,
        //        tmpThree, tmpFour, tmpFive, tmpSix, tmpSeven, tmpEight, tmpNine), BorderLayout.CENTER);
        frame.setVisible(true);
    }
}
