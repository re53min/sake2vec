package org.deeplearning4j.nnpractice;

/**
 * Created by b1012059 on 2015/05/01.
 */

public class BackPropagation {
    //各層の配列(入力層、中間層、出力層)
    private int input[];
    private double hidden[];
    private double output[];
    //各層の重み配列
    private double wIH[][];
    private double wHO[][];
    //中間層、出力層の閾値配列
    private double threshHid[];
    private double threshOut[];
    //中間層、出力層の誤差配列
    private double errorHid[];
    private double errorOut[];
    //学習率
    private final double alpha = 1.0;
    //シグモイド関数の傾き
    private final double beta = 1.0;


    /**
     * コンストラクタ
     * @param INPUT
     * @param HIDDEN
     * @param OUTPUT
     */
    public BackPropagation(int INPUT, int HIDDEN, int OUTPUT){
        int i, j;
        input = new int[INPUT];
        hidden = new double[HIDDEN];
        output = new double[OUTPUT];
        wIH = new double[HIDDEN][INPUT];
        wHO = new double[OUTPUT][HIDDEN];
        threshHid = new double[HIDDEN];
        threshOut = new double[OUTPUT];
        errorHid = new double[HIDDEN];
        errorOut = new double[OUTPUT];

        //入力層→中間層と中間層→出力層の重み配列と閾値配列をランダム(-0.5~0.5)で初期化
        for(i = 0; i < HIDDEN; i++){
            threshHid[i] = Math.random() - 0.5;
            //System.out.println("デバッグポイント1:" + threshHid[i]);
            for(j = 0; j < INPUT; j++){
                wIH[i][j] = Math.random() - 0.5;
                //System.out.println("デバッグポイント2:" + wIH[i][j]);
            }
        }
        for(i = 0; i < OUTPUT; i++){
            threshOut[i] = Math.random() - 0.5;
            //System.out.println("デバッグポイント3:" + threshOut[i]);
            for(j = 0; j < HIDDEN; j++){
                wHO[i][j] = Math.random() - 0.5;
                //System.out.println("デバッグポイント4:" + wHO[i][j]);
            }
        }
    }

    /**
     * BP前向き計算
     */
    public void frontCal(int inputData[]){
        int i,j;
        //各層の長さ
        int lengthIn = input.length;
        int lengthHid = hidden.length;
        int lengthOut = output.length;
        //計算用temp
        double tmpData;

        /*//入力層
        for(i = 0; i < lengthIn; i++){
            input[i] = inputData[i];
        }*/

        //入力層→中間層の計算
        for(i = 0; i < lengthHid; i++){
            tmpData = -threshHid[i];
            for(j = 0; j < lengthIn; j++){
                tmpData = tmpData + input[j] * wIH[i][j];
                //System.out.println("デバッグポイント5:" + tmpData);
            }
            hidden[i] = funSigmoid(tmpData);
            //System.out.println("中間層の値:" + hidden[i]);
        }
        //中間層→出力層の計算
        for(i = 0; i < lengthOut; i++){
            tmpData = -threshOut[i];
            for(j = 0; j < lengthHid; j++){
                tmpData = tmpData + hidden[j] * wHO[i][j];
                //System.out.println("デバッグポイント7:" + tmpData);
            }
            output[i] = funSigmoid(tmpData);
            //System.out.println("出力層の値:" + output[i]);
        }
    }

    /**
     * シグモイド関数
     * @param tmpData
     * @return sigmoid tmpDataを引数としたシグモイト関数の計算結果
     */
    public double funSigmoid(double tmpData){
        double sigmoid = 1.0 / (1.0 + Math.exp(-beta * tmpData));
        return sigmoid;
    }

    /**
     * 教師信号
     * @return t 教師信号
     */
    public double teach(){
        double t;
        //入力層1と入力層2の和が1の場合は「1 - 出力層」、和が0または2の時は「0 - 出力層」
        if((input[0] + input[1]) % 2 == 1 ){
            t = 1.0 - output[0];
        } else {
            t = 0.0 - output[0];
        }
        return t;
    }

    /**
     * 中間層、出力層の誤差計算
     */
    public void errorCal(){
        int i,j;
        //各層の長さ
        int lengthHid = hidden.length;
        int lengthOut = output.length;
        //計算用temp
        double tmpData;

        //出力層の誤差計算
        for(i = 0; i < lengthOut; i++){
            errorOut[i] = teach() * output[i] * (1.0 - output[i]);
            //System.out.println("出力層の誤差:" + errorOut[i]);
        }
        //中間層の誤差計算
        for(i = 0; i < lengthHid; i++){
            tmpData = 0.0;
            for(j = 0; j < lengthOut; j++){
                tmpData = tmpData + errorOut[j] * wHO[j][i];
                //System.out.println("デバッグポイント10:" + tmpData);
            }
            errorHid[i] = hidden[i] * tmpData * (1 - hidden[i]);
            //System.out.println("中間層の誤差:" + errorHid[i]);
        }
    }

    /**
     * BP後ろ向き計算
     */
    public void backCal(){
        int i,j;
        //各層の長さ
        int lengthIn = input.length;
        int lengthHid = hidden.length;
        int lengthOut = output.length;

        //出力層と中間層の学習
        for(i = 0; i < lengthOut; i++){
            threshOut[i] = threshOut[i] - alpha * errorOut[i];
            for(j = 0; j < lengthHid; j++){
                wHO[i][j] = wHO[i][j] + alpha * errorOut[i] * hidden[j];
                //System.out.println("中間層→出力層の重み:" + wHO[i][j]);
            }
        }
        //中間層と入力層の学習
        for(i = 0; i < lengthHid; i++){
            threshHid[i] = threshHid[i] - alpha * errorHid[i];
            for(j = 0; j < lengthIn; j++){
                wIH[i][j] = + wIH[i][j] + alpha * errorHid[i] * input[j];
                //System.out.println("入力層→中間層の重み:" + wIH[i][j]);
            }
        }
    }

    /**
     * 正解データと出力結果との二乗誤差を計算する
     * @param teach
     * @return e 二乗誤差
     */
    public double calcError(double teach[]){
        double e = 0.0;
        int lengthOut = output.length;
        int i;

        for(i = 0; i < lengthOut; i++){
            e = e + Math.pow(teach[i] - output[i], 2.0);
        }
        e *= 0.5;
        return e;
    }

    /**
     * mainメソッド
     * @param args
     */
    public static void main(String[] args){

        System.out.println("デバッグ用１");

        int i, count = 0;

        //入力データ
        int inputData[][] = {
                {0,0},
                {1,0},
                {0,1},
                {1,1}
        };
        //出力データの正解例
        double resultData[][] = {
                {0.0},
                {1.0},
                {1.0},
                {0.0}
        };

        //BackPropagationのコンストラクタの生成
        BackPropagation bp = new BackPropagation(2,5,1);

        //BackPropagationによる学習
        while(true){

            //誤差
            double e = 0.0;

            for(i = 0; i < inputData.length; i++){

                bp.input = inputData[i];
                bp.frontCal(inputData[i]);
                bp.errorCal();
                bp.backCal();

                System.out.println("INPUT:" + inputData[i][0] + "," + inputData[i][1] +
                        " -> " + bp.output[0] + "(" + resultData[i][0] + ")");

                e = e + bp.calcError(resultData[i]);
            }

            count++;
            System.out.println("Error = " + e);
            System.out.println(count + "回目");

            if(e < 0.001 || count == 10000) {
                System.out.println("Error < 0.001");
                System.out.println("学習回数:" + count);
                break;
            }
        }
    }
}
